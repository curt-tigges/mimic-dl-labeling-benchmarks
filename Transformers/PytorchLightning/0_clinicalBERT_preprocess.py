import pandas as pd
from transformers import BertTokenizer
import matplotlib.pyplot as plt

from mimic_constants import *
from cs_utils import *

logger.info('Start: {}'.format(__file__))
logger.info('BERT_MODEL_NAME: {}'.format(BERT_MODEL_NAME))


def print_token_length_info(df, prefix):
    word_cnt = df['length']
    logger.info('{}: Total records: {}'.format(prefix, len(df)))
    logger.info('{}: Average token length: {}'.format(prefix, np.mean(word_cnt)))
    logger.info('{}: Over 512 tokens count: {}'.format(prefix, df[df['length'] > 512]['length'].count()))

############################################
# Load default CAML's training data
############################################
train_df = pd.read_csv('%s/train_50.csv' % MIMIC_3_DIR)
train_df['LABELS'] = train_df['LABELS'].apply(lambda x: x.split(';'))

eval_df = pd.read_csv('%s/dev_50.csv' % MIMIC_3_DIR)
eval_df['LABELS'] = eval_df['LABELS'].apply(lambda x: x.split(';'))

test_df = pd.read_csv('%s/test_50.csv' % MIMIC_3_DIR)
test_df['LABELS'] = test_df['LABELS'].apply(lambda x: x.split(';'))

############################################
# Check the length of Clinical Notes
############################################
print_token_length_info(train_df, 'TRAIN')
print_token_length_info(eval_df, 'EVAL')
print_token_length_info(test_df, 'TEST')

word_cnt = train_df['length']
plt.figure(figsize=[8, 5])
plt.hist(word_cnt, bins=40)
plt.axvline(512, c='r')  # BERT's token size limit
plt.xlabel('Word Count/Clinical Note')
plt.ylabel('# of Occurences')
plt.title("Frequency of Word Counts/Clinical Note")
plt.show()

############################################
# StopWords removing
############################################
if REMOVE_STOP_WORDS:
    logger.info('Remove StopWords(eng) start')

    # if FOR_LOCAL_TEST:
    #     train_df = train_df[0:100]
    #     eval_df = eval_df[0:100]
    #     test_df = test_df[0:100]

    import nltk
    nltk.download('stopwords')

    from nltk.corpus import stopwords
    stop = stopwords.words('english')

    train_df['TEXT'] = train_df['TEXT'].apply(lambda x: (' '.join([item for item in x.split() if item not in stop])))
    train_df['length'] = train_df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)

    eval_df['TEXT'] = eval_df['TEXT'].apply(lambda x: (' '.join([item for item in x.split() if item not in stop])))
    eval_df['length'] = eval_df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)

    test_df['TEXT'] = test_df['TEXT'].apply(lambda x: (' '.join([item for item in x.split() if item not in stop])))
    test_df['length'] = test_df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)

    print_token_length_info(train_df, 'TRAIN')
    print_token_length_info(eval_df, 'EVAL')
    print_token_length_info(test_df, 'TEST')

    word_cnt_no_stopwords = train_df['length']
    plt.figure(figsize=[8, 5])
    plt.hist(word_cnt_no_stopwords, bins=40)
    plt.axvline(512, c='r')  # BERT's token size limit
    plt.xlabel('Word Count/Clinical Note (No StopWords)')
    plt.ylabel('# of Occurences')
    plt.title("Frequency of Word Counts/Clinical Note (No StopWords)")
    plt.show()

    logger.info('Remove StopWords(eng) end')
    if FOR_LOCAL_TEST:
        sys.exit(0)

############################################
# Pickling training data
############################################
train_df.to_pickle(TRAIN_PICKLE)
eval_df.to_pickle(DEV_PICKLE)
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

sys.exit(0)
