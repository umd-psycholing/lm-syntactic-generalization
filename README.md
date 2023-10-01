## LM Syntactic Generalization

### Getting Started
Set up a Python virtual environment and download appropriate dependencies.

```
ENV_NAME=
python3 -m virtualenv $ENV_NAME
source $ENV_NAME/bin/activate
pip install -r requirements.txt
```

### Downloading files to go with GRNNs
Make sure you are in the root directory (not one of the subdirectories) for each of these datasets.

Vocabulary:

```
mkdir data/lm-data
cd data/lm-data
wget https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/vocab.txt
```

Pretrained English model: 

```
mkdir models
cd models
wget https://dl.fbaipublicfiles.com/colorless-green-rnns/best-models/English/hidden650_batch128_dropout0.2_lr20.0.pt
```
