# Historical Text Normalization with Synthetic Data

Historical text normalization is the task of translating historical texts
with inconsistent spelling to consistently spelled texts in the contemporary
standard language.

This repository contains all files needed to:
- use trained models for Early New High German
- train new models for new languages

## Setup
Please install the requirements with the following command:
```
pip install -r requirements.txt
```

After installation, please start Python and run the following
two commands to download the NLTK tokenizer:
```
import nltk
nltk.download('punkt')
```

## Usage
The trained models for Early New High German can be used
with the script `normalize.py`.

The script takes three arguments:
- `--source` (or `-s`): a text file with Early New High German sentences
- `--outfile` (or `-o`): the file to which the normalized sentences are written
- `--config` (or `-c`): a JSON file locating the main model and the fallback model

_Example usage:_
```
python normalize.py -s data/test.fnhd -o test.hyps -c config_normalizer.json
```


## Training
In order to train a new model, you need:
- a target side corpus in standard language
- a JSON file with a list of possible target-to-source permutations

In addition, the source side and target side suffixes may be indicated
using the arguments `-s` and `-t`.
The default suffixes are `-s fnhd` and `-t nhd`.

An example of a valid JSON file with permutations for Early New High German
can be found in `data/config.json`.

_Example training:_
```
bash train_model.sh -s fnhd -t nhd data/sample_corpus.txt data/config.json
```
