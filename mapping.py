import os
import json
import argparse
from collections import OrderedDict
import numpy as np
import time
import torch

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator

EMB_DIR = 'data/fasttext-vectors/'

# main
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
# parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--device", type=str, default="cuda", help="Run on GPU or CPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")

# data
parser.add_argument("--src_langs", type=str, nargs='+', default=['de', 'es', 'fr', 'it', 'pt'], help="Source languages")
parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
# MPSR parameters
parser.add_argument("--mpsr_optimizer", type=str, default="adam", help="Multilingual Pseudo-Supervised Refinement optimizer")
parser.add_argument("--mpsr_orthogonalize", type=bool_flag, default=True, help="During MPSR, whether to perform orthogonalization")
parser.add_argument("--mpsr_n_steps", type=int, default=30000, help="Number of optimization steps for MPSR")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_train", type=str, default="default", help="Path to training dictionary (default or identical_char)")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
parser.add_argument("--semeval_ignore_oov", type=bool_flag, default=True, help="Whether to ignore OOV in SEMEVAL evaluation (the original authors used True)")
# reload pre-trained embeddings
parser.add_argument("--src_embs", type=str, nargs='+', default=[], help="Reload source embeddings (should be in the same order as in src_langs)")
parser.add_argument("--tgt_emb", type=str, default='', help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")


# parse parameters
params = parser.parse_args()

# post-processing options
params.src_N = len(params.src_langs)
params.all_langs = params.src_langs + [params.tgt_lang]
# load default embeddings if no embeddings specified
if len(params.src_embs) == 0:
    params.src_embs = []
    for lang in params.src_langs:
        params.src_embs.append(os.path.join(EMB_DIR, f'wiki.{lang}.vec'))
if len(params.tgt_emb) == 0:
    params.tgt_emb = os.path.join(EMB_DIR, f'wiki.{params.tgt_lang}.vec')

# check parameters
assert not params.device.lower().startswith('cuda') or torch.cuda.is_available()
assert params.dico_train in ["identical_char", "default"] or os.path.isfile(params.dico_train)
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
assert all([os.path.isfile(emb) for emb in params.src_embs])
assert os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
# N+1 embeddings, N mappings , N+1 discriminators
embs, mappings, discriminators = build_model(params, False)
trainer = Trainer(embs, mappings, discriminators, params)
evaluator = Evaluator(trainer)

trainer.reload_best()