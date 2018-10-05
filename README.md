# AttnMeSH

This is the code for [AttentionMeSH: Simple, Effective and Interpretable Automatic MeSH Indexer](https://andy-jqa.github.io/#EMNLP2018), which is published at BioASQ 2018 Workshop of EMNLP 2018.

# Setup

## Prerequisites

- Python 3.6
- torch==0.4.0
- numpy==1.13.3
- Maybe more, just use `pip install` if you run into problems.

## Data

We are considering releasing the preprocessing codes. For now, we just provide small preprocessed training set and test set, with only 50 abstracts each. The toy datasets are located in `data/`. Since the data size is very small, you might observe ovefitting very soon.

## Word Embedding

We use the [pre-trained word embeddings](http://participants-area.bioasq.org/tools/BioASQword2vec/) and the corresponding tokenizer provided by the BioASQ.

After downloading the file, unzip the `biomedicalWordVecotrs.tar.gz` by:

```bash
tar -xvzf biomedicalWordVectors.tar.gz
```

Then you will get a directory `word2vecTools/`. If this directory is not in the `AttnMeSH/`, please specify the path when running the codes.

## MeSH Embedding

We use the MeSH embedding pre-trained by GloVe. The embedding is stored in `mesh_emb/`.

# Run

Use `python run.py` to run AttentionMeSH. You can change the default setting by editting the `config.py`.

```
$ python run.py --help
usage: run.py [-h] [--batch_size BATCH_SIZE] [--seed SEED] [--lr LR]
              [--num_epoch NUM_EPOCH] [--w2v_dir W2V_DIR]
              [--mask_size MASK_SIZE]

Run AttentionMeSH

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size (Default: 8)
  --seed SEED           random seed (Default: 0)
  --lr LR               Adam learning rate (Default: 0.001)
  --num_epoch NUM_EPOCH
                        number of epochs to train (Default: 64)
  --w2v_dir W2V_DIR     The path to pre-trained w2v directory (Default
                        word2vecTools)
  --mask_size MASK_SIZE
                        Size of MeSH mask (Default: 256)
```
