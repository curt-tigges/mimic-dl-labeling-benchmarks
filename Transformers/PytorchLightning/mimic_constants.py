import numpy as np
import torch

############################################
# CAML's preprocessed MIMIC data directory (train_50.csv, dev_50.csv, test_50.csv)
############################################
MIMIC_3_DIR = '/CS598-DLH/caml-mimic/mimicdata/mimic3'

############################################
# Pipeline temp files
############################################
BEST_MODEL_INFO_PATH = './best_model.txt'
BEST_THRESHOLD_INFO_PATH = './best_threshold.txt'
PICKLE_TEST_INPUT_IDS = "./pickle_test_input_ids.pkl"
PICKLE_TEST_ATTENTION_MASKS = "./pickle_test_attention_masks.pkl"
TEST_PREDICT_OUTS = './pickle_test_predict_outs.pkl'
TEST_TRUE_LABELS = './pickle_test_true_labels.pkl'

FOR_LOCAL_TEST = False

############################################
# Model name define
############################################
# BERT_TOKENIZER_DIR = './BERT_tokenizer/'
# BERT_MODEL_NAME = "bert-base-cased"
BERT_TOKENIZER_DIR = './BERT_tokenizer_emilyalsentzer/Bio_ClinicalBERT/'
BERT_MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'

############################################
# Initialize the Hyperparameters
############################################
NUM_WORKERS = 32
N_EPOCHS = 12
BATCH_SIZE = 32
MAX_LEN = 512
LR = 2e-05

############################################
# Etc
############################################
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)