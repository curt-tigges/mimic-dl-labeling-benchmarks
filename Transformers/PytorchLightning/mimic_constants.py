import numpy as np
import torch

MIMIC_3_DIR = '/CS598-DLH/caml-mimic/mimicdata/mimic3'
BEST_MODEL_INFO_PATH = './1_best_model.txt'
BEST_THRESHOLD_INFO_PATH = './2_best_threshold.txt'

FOR_LOCAL_TEST = False

# BERT_TOKENIZER_DIR = './BERT_tokenizer/'
# BERT_MODEL_NAME = "bert-base-cased"
BERT_TOKENIZER_DIR = './BERT_tokenizer_emilyalsentzer/Bio_ClinicalBERT/'
BERT_MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'

CHECKPOINT_PATH = '/workspace/checkpoints/Clinic-epoch=11-val_loss=0.32.ckpt'

############################################
# Initialize the Hyperparameters
############################################
NUM_WORKERS = 32
N_EPOCHS = 12
BATCH_SIZE = 32
MAX_LEN = 512
LR = 2e-05

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)