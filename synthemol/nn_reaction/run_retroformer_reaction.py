from synthemol.constants import RETROFORMER_MODEL_PATH
import os
from synthemol.nn_reaction.retroformer.translate import prepare_retroformer, run_retroformer
from typing import List
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

class rxnretroformer():
    def __init__(
            self,
            topk:int,
            k,
            lam,
            num_beams,
            model_path:str = RETROFORMER_MODEL_PATH,
            
        ):
        self.model_retroformer, self.args_retroformer = prepare_retroformer(
            path=os.path.join(model_path, "model.pt"), 
            vocab_path=os.path.join(model_path, "retroformer_vocab_share.pk"), 
            expansion_topk=topk,
            beam_size=num_beams
        )
        self.k = k
        self.lam = lam

    def forward_prediction(self, smis:List, products_from_smarts:List[List[str]]=None):
        results_retroformer = run_retroformer(
            self.model_retroformer, 
            self.args_retroformer, 
            smis
        )
        products_list = results_retroformer['products']
        sequence_scores_list = results_retroformer['scores']
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

        return topk_products_list

    def _fp(self, mol):
        return _GEN.GetCountFingerprintAsNumPy(mol).astype(np.float32)

    def _select(self, reactant_smiles, scored_beam, k, lam, products_from_smarts:List[str]=None):
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
            qs.append(score)     
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
    