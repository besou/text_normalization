import argparse
import json
import re

from fairseq.models.transformer import TransformerModel
from nltk.tokenize import word_tokenize


class Normalizer():
    def __init__(self, config):
        self.main_model = TransformerModel.from_pretrained(
            config['main_model']['path'],
            checkpoint_file=config['main_model']['checkpoint_file'],
            bpe='sentencepiece',
            sentencepiece_model=config['main_model']['sentencepiece_model'])
        #self.fallback_model = TransformerModel.from_pretrained(
        #    config['main_model']['path'],
        #    checkpoint_file=config['main_model']['checkpoint_file'],
        #    bpe='sentencepiece',
        #    sentencepiece_model=config['main_model']['sentencepiece_model'])
        self.charset = config['charset']

    def remove_invalid_characters(self, text):
        match = re.findall(f'[^{self.charset} <>]', text)
        if match:
            print(f'WARNING: Found invalid character(s) {match} in input:'\
                   '\n{text}\nCharacter(s) will be removed from input.')
            text = re.sub(f'[^{self.charset} <>]', '', text)
            # add test if length is the same as originally
        return text

    def preprocess(self, text):
        text = text.lower()
        text = self.remove_invalid_characters(text)
        text = ' '.join([re.sub(r'(.+)\[([^\[\]]*?)\??\]', r'\1\2', token) for token in text.split()])
        text = ' '.join([re.sub(r'\[([^\[\]]*?)\??\](.+)', r'\1\2', token) for token in text.split()])
        text = re.sub(r'(?<=\S)(<[^<>]*>[^<>]*</[^<>]*>)', r' ## \1', text)
        text = re.sub(r'(<[^<>]*>[^<>]*</[^<>]*>)(?=\S)', r'\1 ## ', text)
        text = ' '.join(word_tokenize(text))
        text = re.sub(r'< ([^<>]*) > ([^<>]*) < (/[^<>]*) >', r'<\1>\2<\3>', text)
        text = re.sub(r'< ([^<>]*) >', r'<\1>', text)
        text = re.sub(r'# #', r'##', text)
        text = re.sub(r'\[ ([^\[\]]*) \]', lambda x: '[' + x.group(1).replace(' ', '@@') + ']', text)

        ignore_tokens = [(index, token)
                  for (index, token) in enumerate(text.split())
                  if token.startswith('<') and token.endswith('>')
                  or token.startswith('#') and token.endswith('#')
                  or token.startswith('[') and token.endswith(']')]

        model_input = ' '.join([token for token in text.split()
                       if not token.startswith('<') and not token.endswith('>')
                       and not token.startswith('#') and not token.endswith('#')
                       and not token.startswith('[') and not token.endswith(']')])

        return model_input, ignore_tokens

    def postprocess(self, model_output, ignore_tokens):
        model_output = model_output.split()
        for index, token in ignore_tokens:
            model_output.insert(index, token)
        model_output = ' '.join(model_output)
        model_output = re.sub(' ## ', '', model_output)
        model_output = re.sub(r' +(?=[\.,;:!\?\)“’])', '', model_output)
        model_output = re.sub(r'(?<=[„\(]) +', '', model_output)
        model_output = re.sub(r'@@', ' ', model_output)
        return model_output

    def normalize(self, text):
        model_input, ignore_tokens = self.preprocess(text)
        model_output = self.main_model.translate(model_input)
        if len(model_output.split()) != len(model_input.split()):
            print(f'Fallback model is used for:\n{text}')
            model_output = "FAIL." # self.fallback_model.translate(model_input)
        model_output = self.postprocess(model_output, ignore_tokens)
        return model_output

def parse_args():
    parser = argparse.ArgumentParser(description='Normalize text.')
    parser.add_argument('--source', '-s', type=str, required=True,
                        help='Source file with sequences to normalize.')
    parser.add_argument('--outfile', '-o', type=str, required=True,
                        help='File to which the normalized sequences are written.')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='JSON file indicating the parameters for the main model'\
                        'and the fallback model.')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    normalizer = Normalizer(config)

    num_sents = 0

    with open(args.source) as infile, open(args.outfile, 'w') as outfile:
        for line in infile:
            num_sents += 1
            normalized = normalizer.normalize(line.strip())
            outfile.write(normalized+'\n')
            if num_sents % 1000 == 0:
                print(f'Processed {num_sents} sentences.\r', end='')

    print(f'Processed {num_sents} sentences.')

if __name__ == '__main__':
    main()
