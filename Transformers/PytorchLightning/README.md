# Multi-label Text Classification using Transformers(BERT)
Using Transformers pre-trained model for medical code predictions using MIMIC III Clinical notes data

* Data preprocessing based on CAML: https://github.com/jamesmullenbach/caml-mimic
* PyTorch Lightning code based on: [blog](https://medium.com/analytics-vidhya/multi-label-text-classification-using-transformers-bert-93460838e62b), [code](https://github.com/pnageshkar/NLP/blob/master/Medium/Multi_label_Classification_BERT_Lightning.ipynb)

# Requirements

## Set path in mimic_constants.py
   
    # CAML's preprocessed data
    MIMIC_3_DIR = '/CS598-DLH/caml-mimic/mimicdata/mimic3'
   
## Conda environment

    conda create -n multiLabelBERT python=3.7.1 --yes
    conda activate multiLabelBERT
    cat <<EOF | sudo tee ./requirements.txt
    pytorch-lightning==1.2.8
    transformers==4.5.1
    pandas==1.2.4
    scikit-learn==0.24.2
    seaborn==0.11.1
    nltk==3.6.1
    torch==1.8.1
    EOF
    
    pip install -r requirements.txt
    (for A100 / CUDA 11.0) pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    
    * Some packages in conda channel has different name. e.g.) "sklearn" => "scikit-learn"
 
# How to use

# Preprocessing
    python 0_clinicalBERT_preprocess.py

# Train
    python 1_clinicalBERT_train.py

# Find the best threshold
    python 2_clinicalBERT_threshold_find.py

# Test
    python 3_clinicalBERT_test.py