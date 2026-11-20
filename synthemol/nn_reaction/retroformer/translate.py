import math
from time import time
import numpy as np
import pickle
from easydict import EasyDict

import os
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

import sys
root = os.path.abspath('.')
if root not in sys.path:
    sys.path.insert(0, root)

from synthemol.nn_reaction.retroformer.utils.smiles_utils import *
from synthemol.nn_reaction.retroformer.utils.translate_utils import translate_batch_stepwise
from synthemol.nn_reaction.retroformer.utils.build_utils import build_model, load_checkpoint, build_batch

from rdkit import Chem
    

def filter_invalid(generations, scores):
    filtered_generations, filtered_scores = [], []
    for generation, score in zip(generations, scores):
        mol = Chem.MolFromSmiles(generation)
        if mol is None:
            continue
        filtered_generations.append(Chem.MolToSmiles(mol))
        filtered_scores.append(score)
    
    return filtered_generations, filtered_scores


def filter_single_atom(generations):
    filtered_generations = []
    for generation in generations:
        filtered_generations.append('.'.join([smi for smi in generation.split('.') if len(smi) > 1]))
    
    return filtered_generations


def translate(args, batch, models, dataset, singlestep=True):
    #generations, scores = [], []
    invalid_token_indices = [dataset.tgt_stoi['<RX_{}>'.format(i)] for i in range(1, 11)]
    invalid_token_indices += [dataset.tgt_stoi['<UNK>'], dataset.tgt_stoi['<unk>']]

    # Ensure models is always a list (caller may pass a single model)
    if not isinstance(models, list):
        models = [models]

    #n_smiles = len(batch)  # assumes batch length == number of input SMILES
    #generation_dicts = [defaultdict(int) for _ in range(n_smiles)]
    #counters = [defaultdict(int) for _ in range(n_smiles)]
    
    generation_dicts = None
    counters = None

    #generation_dict, counter = defaultdict(int), defaultdict(int)
    for model in models:
        pred_tokens, pred_scores, predicts = \
            translate_batch_stepwise(model, batch, invalid_token_indices=invalid_token_indices,
                                    beam_size=args.beam_size,
                                    T=10, alpha_atom=-1, alpha_bond=-1,
                                    beta=0.5, percent_aa=40, percent_ab=40, k=3,
                                    use_template=False,
                                    factor_func=dataset.factor_func,
                                    reconstruct_func=dataset.reconstruct_smi)
        #print(f"[DEBUG] len(predicts)={len(predicts)}, type(predicts[0])={type(predicts[0])}, len(predicts[0])={len(predicts[0])}")
        #print(f"[DEBUG] pred_tokens.shape={pred_tokens.shape}")
        
        #print(f"[DEBUG] predicts structure: {[(i, len(p)) for i, p in enumerate(predicts)]}")

        if not singlestep:
            assert len(predicts) == 1

        original_beam_size = pred_tokens.shape[1]
        current_i = 0

        #generation_dicts = [defaultdict(int) for _ in range(len(predicts))]
        #counters = [defaultdict(int) for _ in range(len(predicts))]

        if generation_dicts is None:
            n_smiles = len(predicts)
            generation_dicts = [defaultdict(int) for _ in range(n_smiles)]
            counters = [defaultdict(int) for _ in range(n_smiles)]

        original_beam_size = pred_tokens.shape[1]
        current_i = 0

        #print(f"[DEBUG2] isinstance check: {isinstance(models, list)}, len(models)={len(models)}")
        #print(f"[DEBUG2] len(predicts)={len(predicts)}, len(generation_dicts)={len(generation_dicts)}")
        #print(f"[DEBUG2] predicts structure: {[(i, len(p)) for i, p in enumerate(predicts)]}")

        for smi_idx, predict in enumerate(predicts):
        #for input_idx, predict in enumerate(predicts):
            remain = original_beam_size
            beam_size = math.ceil(original_beam_size / len(predict))

            hypo_i, hypo_scores_i = [], []
            for j, (rc, rc_score) in enumerate(predict):
                pred_token = pred_tokens[current_i + j]

                sub_hypo_candidates, sub_score_candidates = [], []
                for k in range(pred_token.shape[0]):
                    hypo_smiles_k = ''.join(dataset.reconstruct_smi(pred_token[k], src=False))
                    hypo_lens_k = len(smi_tokenizer(hypo_smiles_k))
                    hypo_scores_k = pred_scores[current_i + j][k].cpu().numpy() / hypo_lens_k + rc_score / 10

                    if hypo_smiles_k not in hypo_i:  # only select unique entries
                        sub_hypo_candidates.append(hypo_smiles_k)
                        sub_score_candidates.append(hypo_scores_k)

                ordering = np.argsort(sub_score_candidates)[::-1]
                sub_hypo_candidates = list(np.array(sub_hypo_candidates)[ordering])[:min(beam_size, remain)]
                sub_score_candidates = list(np.array(sub_score_candidates)[ordering])[:min(beam_size, remain)]

                hypo_i += sub_hypo_candidates
                hypo_scores_i += sub_score_candidates

                remain -= beam_size

            current_i += len(predict)
            ordering = np.argsort(hypo_scores_i)[::-1]
            #generations.append(np.array(hypo_i)[ordering][:10])
            #scores.append(np.array(hypo_scores_i)[ordering][:10])
        
        #if singlestep:
        #    generations = [list(g) for g in generations]
        #    scores = [list(s) for s in scores]
        #    for i in range(len(scores)):
        #        if len(scores[i]) > 0:
        #            scores[i] = list(np.exp(np.array(scores[i])) / np.sum(np.exp(np.array(scores[i]))))     # softmax
        #    return {'products': generations, 'scores': scores}

            _generations = list(np.array(hypo_i)[ordering])
            _scores = list(np.array(hypo_scores_i)[ordering])

            _generations, _scores = filter_invalid(_generations, _scores)
            _generations = filter_single_atom(_generations)

            #for i, (g, s) in enumerate(zip(_generations, _scores)):
                #generation_dict[g] += s
                #counter[g] += 1
            
            for g, s in zip(_generations, _scores):
                generation_dicts[smi_idx][g] += s
                counters[smi_idx][g] += 1

    #for g, cnt in counter.items():
    #    generation_dict[g] /= cnt
    all_generations, all_scores = [], []
    for smi_idx in range(n_smiles):
        for g, cnt in counters[smi_idx].items():
            generation_dicts[smi_idx][g] /= cnt

        #generations = list(generation_dict.keys())
        #scores = list(generation_dict.values())
        generations = list(generation_dicts[smi_idx].keys())
        scores = list(generation_dicts[smi_idx].values())
    
        #ordering = np.argsort(scores)[::-1][:args.expansion_topk]
        #generations = list(np.array(generations)[ordering])
        #scores = list(np.array(scores)[ordering])
        ordering = np.argsort(scores)[::-1][:args.expansion_topk]
        generations = list(np.array(generations)[ordering])
        scores = list(np.array(scores)[ordering])

        if len(scores) > 0:
            scores = list(np.exp(np.array(scores)) / np.sum(np.exp(np.array(scores))))   # softmax

        all_generations.append(generations)
        all_scores.append(scores)
    
    return {'products': all_generations, 'scores': all_scores}


def prepare_retroformer(path, vocab_path, cuda=True, beam_size=10, expansion_topk=10,
                        ):
    args = EasyDict()
    args.device = 'cuda' if cuda else 'cpu'
    args.beam_size = beam_size
    args.expansion_topk = expansion_topk
    args.vocab = vocab_path
    args.checkpoint = path
    
    with open(args.vocab, 'rb') as f:
        src_itos, tgt_itos = pickle.load(f)

    # ensemble
    args.checkpoint = args.checkpoint.split(',')
    models = [build_model(args, src_itos, tgt_itos) for _ in range(len(args.checkpoint))]
    models = load_checkpoint(args, models)
    return models, args


def run_retroformer(model, args, smi):
    singlestep = True
    if type(smi) == str:
        smi = [smi]
        #singlestep = False
    batch, dataset = build_batch(args, smi) 
    #print(f"[DEBUG] type(batch)={type(batch)}, batch keys (if dict)={batch.keys() if hasattr(batch, 'keys') else 'N/A'}") 
    #print(f"[DEBUG] type(dataset)={type(dataset)}") 
    #print(f"[DEBUG] smi input={smi}")

    result = translate(args, batch, model, dataset, singlestep)
    return result


if __name__ == '__main__':
    t_start = time()
    PATH = '/media/scratch/sonha/projects/pnpgen/PNPGen2/pretrained/retroformer/model.pt'
    VOCAB_PATH = '/media/scratch/sonha/projects/pnpgen/PNPGen2/pretrained/retroformer/retroformer_vocab_share.pk'

    model, args = prepare_retroformer(path=PATH, vocab_path=VOCAB_PATH)
    #smi = 'C=CC(C)(C)[C@@]12C[C@H]3c4nc5ccccc5c(=O)n4[C@H](C)C(=O)N3C1N(C(C)=O)c1ccccc12'
    #smi = 'N[C@@H](CNC(=O)C(=O)O)C(=O)O'
    smi = ['C=CC(C)(C)[C@@]12C[C@H]3c4nc5ccccc5c(=O)n4[C@H](C)C(=O)N3C1N(C(C)=O)c1ccccc12', 'N[C@@H](CNC(=O)C(=O)O)C(=O)O']
    #smi = "C[C@@]1(CC(=O)O)C2=N/C(=C\c3[nH]c(c(CCC(=O)O)c3CC(=O)O)Cc3[nH]c(c(CC(=O)O)c3CCC(=O)O)[C@](C)(O)[C@@]34N/C(=C2)[C@@H](CCC(=O)O)[C@]3(C)CC(=O)O4)[C@H]1CCC(=O)O"
    print(run_retroformer(model, args, smi))
    print(f'\033[92mTotal {time() - t_start:.2f} sec elapsed\033[0m')
