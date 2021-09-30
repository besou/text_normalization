"""
Normalize text with pretrained models.

The script guarantees a one-to-one alignment between an input string
and its normalization. This is done using two different models:
The main model translates full sequences. If it fails to generate a one-to-one
alignment between the input and the output, the fallback model is used
to generate a normalization of each word separately.

The script can handle XML tags and will simply skip them during normalization.
"""

import argparse
import json
import re

from fairseq.models.transformer import TransformerModel
from nltk.tokenize import word_tokenize


class Normalizer():
    """
    Class to normalize strings with pretrained models.

    Args:
        config (str): path to a JSON configuration file.

    Attributes:
        main_model: the main model which translates full sequences.
        fallback_model: the fallback model used when the main model
            fails to generate a one-to-one alignment.
        charset (str): set of valid characters in the input.
    """

    def __init__(self, config):
        self.main_model = TransformerModel.from_pretrained(
            config['main_model']['path'],
            checkpoint_file=config['main_model']['checkpoint_file'],
            bpe='sentencepiece',
            sentencepiece_model=config['main_model']['sentencepiece_model'])
        self.fallback_model = TransformerModel.from_pretrained(
            config['fallback_model']['path'],
            checkpoint_file=config['fallback_model']['checkpoint_file'],
            bpe='sentencepiece',
            sentencepiece_model=config['fallback_model']['sentencepiece_model'])
        self.charset = config['charset']

    def remove_invalid_characters(self, text):
        """
        Remove all characters from a string that are not in self.charset.

        Args:
            text (str): The string to be processed.

        Returns:
            str: The processed string.
        """

        match = re.findall(f'[^{self.charset} <>]', text)
        if match:
            print(f'WARNING: Found invalid character(s) {match} in input:'\
                   '\n{text}\nCharacter(s) will be removed from input.')
            text = re.sub(f'[^{self.charset} <>]', '', text)
            # TODO: add test if length is the same as originally
        return text

    def preprocess(self, text):
        """
        Preprocess a string:
        - lowercase everything
        - remove invalid characters
        - tokenize
        - remove text between tags (<>) and square brackets ([]),
            but remember their index

        Args:
            text (str): A string to preprocess.

        Returns:
            tuple: the preprocessed string, a list with text
                between tags and square brackets
        """

        text = text.lower()
        text = self.remove_invalid_characters(text)
        text = ' '.join([re.sub(r'(.+)\[([^\[\]]*?)\??\]', r'\1\2', token)
                         for token in text.split()])
        text = ' '.join([re.sub(r'\[([^\[\]]*?)\??\](.+)', r'\1\2', token)
                         for token in text.split()])
        text = re.sub(r'(?<=\S)(<[^<>]*>[^<>]*</[^<>]*>)', r' ## \1', text)
        text = re.sub(r'(<[^<>]*>[^<>]*</[^<>]*>)(?=\S)', r'\1 ## ', text)
        text = ' '.join(word_tokenize(text))
        text = re.sub(r'< ([^<>]*) > ([^<>]*) < (/[^<>]*) >',
                      r'<\1>\2<\3>', text)
        text = re.sub(r'< ([^<>]*) >', r'<\1>', text)
        text = re.sub(r'# #', r'##', text)
        text = re.sub(r'\[ ([^\[\]]*) \]',
                      lambda x: '[' + x.group(1).replace(' ', '@@') + ']',
                      text)

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
        """
        Postprocess model output:
        - re-introduce tags and square brackets
        - detokenize

        Args:
            model_output: A normalized string.
            ignore_tokens: A list with ignored tokens (text in tags
                and in square brackets) and their index.

        Returns:
            The string with the ignored tokens inserted at their index.
        """

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
        """
        Normalize a string:
        - preprocess
        - normalize
        - postprocess

        Args:
            text (str): A string to normalize.

        Returns:
            str: The normalized string.
        """

        model_input, ignore_tokens = self.preprocess(text)

        # use main model
        model_output = self.main_model.translate(model_input)

        # use fallback model if main model failed
        if len(model_output.split()) != len(model_input.split()):
            print(f'Fallback model is used for:\n{text}')
            model_output = []
            text = model_input.strip().lower().split()
            text = ['<pad>']*5 + [word.strip() for word in text] + ['<pad>']*5
            for i in range(len(text)-10):
                current = ' '.join(text[i:i+5]) + f' <token> {text[i+5]} </token> ' + ' '.join(text[i+6:i+11])
                print(current)
                prediction = self.fallback_model.translate(current)
                print(prediction)
                model_output.append(prediction)
            model_output = ' '.join(model_output)

        model_output = self.postprocess(model_output, ignore_tokens)
        return model_output

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description='Normalize text.')
    parser.add_argument('--source', '-s', type=str, required=True,
                        help='Source file with sequences to normalize.')
    parser.add_argument('--outfile', '-o', type=str, required=True,
                        help='File to which the normalized sequences '\
                        'are written.')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='JSON file indicating the parameters for the '\
                        'main model and the fallback model.')
    return parser.parse_args()

def main():
    """
    Initialize a normalizer and process an input file with it.
    """

    args = parse_args()

    with open(args.config) as jsonfile:
        config = json.load(jsonfile)

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
