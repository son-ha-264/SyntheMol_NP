from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rdkit import Chem
import pandas as pd
from typing import List

import numpy as np
from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

import torch

RDLogger.DisableLog("rdApp.*")
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

class rxnt5():
    def __init__(self, device, path_t5, k=3, lam=0.65):
        self.tokenizer = AutoTokenizer.from_pretrained(path_t5, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path_t5, trust_remote_code=True).to(device).eval()
        self.device = device
        self.k = k
        self.lam = lam

    def forward_prediction(self, smis, products_from_smarts:List[List[str]]=None, topk=10, num_beams=10):
        """Run ReactionT5 model to retrieve products"""
        assert type(smis) == list
        assert type(smis[0]) == str

        input_t5 = ['REACTANT:' + r.split('>>')[0] for r in smis]
        inp = self.tokenizer(input_t5, return_tensors='pt', padding=True)
        inp = inp.to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inp, num_beams=num_beams, num_return_sequences=num_beams, return_dict_in_generate=True, output_scores=True)
        products_list = [self.tokenizer.decode(output['sequences'][i], skip_special_tokens=True).replace(' ', '').rstrip('.') for i in range(len(smis)*topk)] 
        products_list = [products_list[i:i+topk] for i in range(0, len(products_list), topk)]
        sequence_scores_list = output.sequences_scores.tolist()
        sequence_scores_list = [output.sequences_scores[i] for i in range(len(smis)*topk)] 
        sequence_scores_list = [sequence_scores_list[i:i+topk] for i in range(0, len(sequence_scores_list), topk)]

        #products_list = self._remove_similar_stereo_and_invalid(products_list)
        #topk_products_list = [products_list[i*topk:(i+1)*topk] for i in range(len(smis))]
        topk_products_list = []

        assert len(smis) == len(products_list) 
        assert len(smis) == len(sequence_scores_list)

        if products_from_smarts:
            for smarts_prod, reactant_smiles, prods, scores in zip(products_from_smarts, smis, products_list, sequence_scores_list):
                scored_beam = list(zip(prods, scores))
                topk_products_list.append(self._select(reactant_smiles=reactant_smiles, scored_beam=scored_beam, products_from_smarts=smarts_prod, k=self.k, lam=self.lam))
        else:
            for reactant_smiles, prods, scores in zip(smis, products_list, sequence_scores_list):
                scored_beam = list(zip(prods, scores))
                topk_products_list.append(self._select(reactant_smiles=reactant_smiles, scored_beam=scored_beam, products_from_smarts=None, k=self.k, lam=self.lam))            

        #return {'products': topk_products_list}
        return topk_products_list

    """
    def _remove_similar_stereo_and_invalid(self, list_smis):

        mols = [Chem.MolFromSmiles(i) for i in list_smis]
        mols = [r for r in mols if r is not None]

        list_smis = [Chem.MolToSmiles(mol) for mol in mols]

        df_smi = pd.DataFrame(
            {
                "smi": list_smis,
                "smi_no_stereo":[self._strip_stereo(smi) for smi in list_smis]
            }
        )

        df_smi = df_smi.drop_duplicates("smi_no_stereo")
        return df_smi['smi'].to_list()


    def _strip_stereo(self, smi):
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, isomericSmiles=False)"""

    def _fp(self, mol):
        return _GEN.GetCountFingerprintAsNumPy(mol).astype(np.float32)

    def _select(self, reactant_smiles, scored_beam, products_from_smarts:List[str]=None, k=3, lam=0.0):
        """beam: [(product_smiles, logprob)]
        One reaction with multiple possible reactants
        """
        r_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles.split(".")]
        r_fp = sum(self._fp(m) for m in r_mols)
        r_set = {Chem.MolToSmiles(m) for m in r_mols}

        # List smiles, list quality score, list difference between fp of product and reactant
        smis, qs, deltas = [], [], [] 
        seen_flat = set()     

        for smi, score in sorted(scored_beam, key=lambda x: -x[1]): 

            # Check if it matches the products from the smarts string
            if products_from_smarts and smi in products_from_smarts:
                continue
            
            mol = Chem.MolFromSmiles(smi)
            # Validity check
            if mol is None:
                continue

            # Remove products similar to reactant
            can = Chem.MolToSmiles(mol)
            if can in r_set or can in smis:                     
                continue        

            # Remove products which only different in stereochemistry
            flat = Chem.MolToSmiles(mol, isomericSmiles=False)  
            if flat in seen_flat:                      
                continue

            seen_flat.add(flat)               
            smis.append(can)
            qs.append(score.detach().cpu().numpy())     
            deltas.append(self._fp(mol) - r_fp)

        if not smis:
            return []


        q = np.array(qs)
        q = (q - q.min()) / (np.ptp(q) + 1e-9)
        D = np.stack(deltas)
        D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
        S = np.clip(D @ D.T, 0, 1)

        # Choose k products with Maximal Marginal Relevance
        # Pick candidate greedily, with score being lamb*quality + (1-lamb)*(max sim to what is already picked)
        picked = [int(q.argmax())]
        while len(picked) < min(k, len(smis)):
            score = lam * q - (1 - lam) * S[:, picked].max(axis=1)
            score[picked] = -np.inf
            picked.append(int(score.argmax()))

        return [smis[i] for i in picked]        


if __name__ =="__main__":

    import time
    reactions_list = [
        "CC(C)(C)OC(=O)N1CCC(C)(C)C(N)C1.CCCc1ccc(C(=O)O)[nH]1.O=C(O)c1cc(=O)c2ccccc2[nH]1>>CCCc1ccc(C(=O)NC2CN(C(=O)c3cc(=O)c4ccccc4[nH]3)CCC2(C)C)[nH]1",
        "CC(C)(C)OC(=O)N(CCO)C1CCNC1.Cc1c(C(=O)O)nnn1Cc1ccncc1.O=C(O)c1ccc(C2CCOC2)s1>>Cc1c(C(=O)N2CCC(N(CCO)C(=O)c3ccc(C4CCOC4)s3)C2)nnn1Cc1ccncc1",
        "CC(C)(C)OC(=O)N1CCCC12CCC(N)CC2.CC(C)c1cc(C(=O)O)on1.O=C(O)C1CC2COCC(C1)C2(F)F>>CC(C)c1cc(C(=O)NC2CCC3(CCCN3C(=O)C3CC4COCC(C3)C4(F)F)CC2)on1",
        "CC(C)(C)OC(=O)N[C@@H]1C[C@@]2(N)CO[C@@H]1C2.CC(C)CC1(C(=O)O)CC1.O=C(O)c1cnc2cc[nH]c2c1>>CC(C)CC1(C(=O)N[C@]23CO[C@H](C2)[C@H](NC(=O)c2cnc4cc[nH]c4c2)C3)CC1",
        "CNCC1CN(C(=O)OC(C)(C)C)CCO1.CCC(C)C(OC)C(=O)O.CC(C)(C)OC(=O)NC1(C(=O)O)COC(C)(C)OC1>>CCC(C)C(OC)C(=O)N(C)CC1CN(C(=O)C2(NC(=O)OC(C)(C)C)COC(C)(C)OC2)CCO1",
        "CC(C)(C)OC(=O)N1CC(O)(CN)C1.CCC(C)n1ncc(C(=O)O)c1C1CC1.Cn1ncc(C(=O)O)c1C1CC1>>CCC(C)n1ncc(C(=O)NCC2(O)CN(C(=O)c3cnn(C)c3C3CC3)C2)c1C1CC1",
    ]

    reactant_smiles = [r.split(">>")[0] for r in reactions_list]
    products_from_smarts = [r.split(">>")[1] for r in reactions_list]

    rxnt5_func = rxnt5(device='cuda:0', path_t5="/home/sonh/MAINCE_project/SyntheMol_NP/pretrained/rt5_ft_uspto_npl_custom_replay01_kl01/best")

    t0 = time.time()
    res = rxnt5_func.forward_prediction(smis=reactant_smiles, products_from_smarts=products_from_smarts, topk=10, num_beams=10)
    t1 = time.time()

    print(res)
    print(f"Time taken: {t1-t0}")

    #CUDA_VISIBLE_DEVICES=GPU-49d7e714-4862-c87f-9399-41201b8a3d15 python synthemol/nn_reaction/run_t5_reaction.py 
    #CUDA_VISIBLE_DEVICES=0 python synthemol/nn_reaction/run_t5_reaction.py 