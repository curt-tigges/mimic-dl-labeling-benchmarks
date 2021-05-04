# Multi-label Text Classification using Transformers(BERT)
Using Transformers pre-trained model for medical code predictions using MIMIC III Clinical notes data

* Data preprocessing based on CAML: https://github.com/jamesmullenbach/caml-mimic
* PyTorch Lightning code based on: [blog](https://medium.com/analytics-vidhya/multi-label-text-classification-using-transformers-bert-93460838e62b), [code](https://github.com/pnageshkar/NLP/blob/master/Medium/Multi_label_Classification_BERT_Lightning.ipynb)

# Requirements

## CAML's preprocessed data:
   
    MIMIC_3_DIR = '/CS598-DLH/caml-mimic/mimicdata/mimic3'
   
## Conda environment

    conda create -n multiLabelBERT python=3.7.1
    conda activate multiLabelBERT
    pip install -q pytorch-lightning
    pip install -q bs4
    pip install -q transformers
    pip install pandas
    pip install sklearn
    pip install seaborn
    pip install pylab

