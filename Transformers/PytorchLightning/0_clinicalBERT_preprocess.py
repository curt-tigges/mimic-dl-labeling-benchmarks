import pandas as pd
from transformers import BertTokenizer

from mimic_constants import *
from cs_utils import *

############################################
# Pickling training data
############################################
train_df = pd.read_csv('%s/train_50.csv' % MIMIC_3_DIR)
train_df['LABELS'] = train_df['LABELS'].apply(lambda x: x.split(';'))
train_df.to_pickle(TRAIN_PICKLE)

eval_df = pd.read_csv('%s/dev_50.csv' % MIMIC_3_DIR)
eval_df['LABELS'] = eval_df['LABELS'].apply(lambda x: x.split(';'))
eval_df.to_pickle(DEV_PICKLE)

test_df = pd.read_csv('%s/test_50.csv' % MIMIC_3_DIR)
test_df['LABELS'] = test_df['LABELS'].apply(lambda x: x.split(';'))
test_df.to_pickle(TEST_PICKLE)

############################################
# Pickling test data's BERT input
############################################
logger.info('BERT tokenizer load start')
Bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
logger.info('BERT tokenizer load end')

logger.info('Setup test dataset for BERT start')
input_ids = []
attention_masks = []
x_test = test_df['TEXT']
for note in x_test:
    text_enc = Bert_tokenizer.encode_plus(
        note,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_token_type_ids=False,
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt'
    )

    # Add the input_ids from encoded ClinicalNote to the list.
    input_ids.append(text_enc['input_ids'])
    # Add its attention mask
    attention_masks.append(text_enc['attention_mask'])
logger.info('Setup test dataset for BERT end')

with open(PICKLE_TEST_INPUT_IDS, 'wb') as f:
    pickle.dump(input_ids, f)

with open(PICKLE_TEST_ATTENTION_MASKS, 'wb') as f:
    pickle.dump(attention_masks, f)

