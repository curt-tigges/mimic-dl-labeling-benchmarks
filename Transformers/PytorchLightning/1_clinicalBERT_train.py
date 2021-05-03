from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer

from mimic_models import *
from mimic_constants import *
from cs_utils import *

############################################
# 1. Load & Pre-process the data
############################################
train_df = load_pickle('{}/train.pkl'.format(MIMIC_3_DIR))
eval_df = load_pickle('{}/dev.pkl'.format(MIMIC_3_DIR))
test_df = load_pickle('{}/test.pkl'.format(MIMIC_3_DIR))

x_tr = train_df['TEXT']
x_val = eval_df['TEXT']
x_test = test_df['TEXT']

logger.info('load data end')


############################################
# 2.1 Encoding label
############################################
mlb = MultiLabelBinarizer()
y_tr = mlb.fit_transform(train_df['LABELS'])  # y tag(label)
y_val = mlb.fit_transform(eval_df['LABELS'])  # y tag(label)
y_test = mlb.fit_transform(test_df['LABELS'])  # y tag(label)

logger.info(mlb.classes_)
logger.info('label encoding end')


############################################
# 2.2 Check the length of Clinical Notes
############################################
# word_cnt = train_df.pop('length')
# plt.figure(figsize=[8, 5])
# plt.hist(word_cnt, bins=40)
# plt.xlabel('Word Count/Clinical Note')
# plt.ylabel('# of Occurences')
# plt.title("Frequency of Word Counts/Clinical Note")
# plt.show()


############################################
# 3. Define our model
############################################
# Initialize the Bert tokenizer
logger.info('BERT tokenizer load start')
Bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
logger.info('BERT tokenizer load end')

# Instantiate and set up the data_module
data_module = MimicDataModule(x_tr, y_tr, x_val, y_val, x_test, y_test, Bert_tokenizer, BATCH_SIZE, MAX_LEN)
data_module.setup()


############################################
# 4. Train the model
############################################
# Instantiate the classifier model
steps_per_epoch = len(x_tr) // BATCH_SIZE
model = MimicClassifier(n_classes=50, steps_per_epoch=steps_per_epoch, n_epochs=N_EPOCHS, lr=LR)

# Initialize Pytorch Lightning callback for Model checkpointing
# saves a file like: input/MIMIC-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # monitored quantity
    filename='MIMIC-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,  # save the top 3 models
    mode='min',  # mode of the monitored quantity for optimization
)

# Instantiate the Model Trainer
if torch.cuda.is_available():
    # for GPUs
    trainer = pl.Trainer(max_epochs=N_EPOCHS, gpus=1, accelerator='ddp',
                         callbacks=[checkpoint_callback], progress_bar_refresh_rate=30)
else:
    # for CPUs
    trainer = pl.Trainer(max_epochs=N_EPOCHS, gpus=0,
                         callbacks=[checkpoint_callback], progress_bar_refresh_rate=30)

# Train the Classifier Model
trainer.fit(model, data_module)

logger.info("train end")

# Retrieve the checkpoint path for best model
model_path = checkpoint_callback.best_model_path
logger.info('Best model checkpoint path: {}'.format(model_path))

with open(BEST_MODEL_INFO_PATH, 'w') as f:
    f.write(model_path)


