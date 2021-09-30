"""
Clean the target-side corpus:
- remove bad samples from training data
- restrict character range
- tokenize
- optional: prepare n-to-m mapping
"""

from random import random, randint
import re
import sys

import json
from nltk.tokenize import word_tokenize
from langid.langid import LanguageIdentifier, model
import pyphen
from unidecode import unidecode


class TargetDataCleaner():
    """
    Class to clean target-side data.

    Args:
        config (str): path to a JSON configuration file.

    Attributes:
        charset (str): set of valid characters in the input.
        lang (str): two-letter language code.
        n_to_m (bool): option to allow n-to-m mapping between source and target.
        n_copies (int): number of times each sample is copied
            in order to enlarge the corpus.
        identifier: model to identify the language of a string.
        dic: Pyphen class to syllable-tokenize a string.
    """

    def __init__(self, config):
        self.charset = config['charset_target']
        self.lang = config['lang']
        self.n_to_m = config['n_to_m']
        self.n_copies = config['num_copies']
        self.identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        self.identifier.set_languages(config['filter_langs'].append(self.lang))
        self.dic = pyphen.Pyphen(lang='de_DE', left=1, right=1)


    def remove_chars(self, text):
        """
        Normalize punctuation.

        Args:
            text (str): a string.

        Returns:
            str: the string with normalized punctuation.
        """

        text = text.replace(r'[_\+\*$#=~\[\]\{\}]', '')
        text = re.sub(r'(- ?)+', '-', text)
        text = re.sub('—', '-', text)
        text = re.sub(rf' [«‹»›](?=[{self.charset}])', '„', text)
        text = re.sub(rf'(?<=[{self.charset}])[«‹»›] ', '“', text)
        return text


    def restrict_charset(self, text):
        """
        Replace all non-ASCII characters that are not in self.charset
        with their closest ASCII equivalent.

        Args:
            text (str): a string.

        Returns:
            str: the string with normalized character range.
        """

        exceptions = set(c for c in self.charset if len(c.encode()) > 1)
        text = ''.join([unidecode(char) if len(char.encode()) > 1
            and char not in exceptions else char for char in text])
        return text


    def word_tokenize(self, text):
        """
        Tokenize a string.

        Args:
            text (str): a string.

        Returns:
            str: the tokenized string.
        """

        text = word_tokenize(text)
        return ' '.join(text)


    def _find_nth(self, string, substring, n):
        if n == 1:
            return string.find(substring)
        return string.find(substring, self._find_nth(string, substring, n-1) + 1)


    def prepare_n_to_m_mapping(self, text):
        """
        Merge and split some tokens to enable n-to-m mapping.

        Args:
            text (str): a string.

        Returns:
            str: the processed string.
        """

        if random() > 0.99 and text.count(' ') > 2:
            n_spaces = text.count(' ')
            n = randint(1, n_spaces-1)
            position = self._find_nth(text, ' ', n)
            text = list(text)
            text[position] = '_'
            text = ''.join(text)
        if random() > 0.99:
            text = text.split()
            position = randint(0, len(text)-1)
            split = self.dic.wrap(text[position], 6)
            if split is not None:
                text[position] = ' '.join(split)
            text = ' '.join(text)
        return text


    def preprocess(self, text):
        """
        Preprocess a string:
        - normalize punctuation
        - restrict character set
        - tokenize
        - optional: prepare potential n-to-m mapping

        Args:
            text (str): a string.

        Returns:
            str: the preprocessed string.
        """

        text = self.remove_chars(text)
        text = self.restrict_charset(text)
        text = self.word_tokenize(text)
        text = re.sub(r' +', ' ', text)
        texts = []
        for _ in range(self.n_copies):
            if self.n_to_m:
                texts.append(self.prepare_n_to_m_mapping(text))
            else:
                texts.append(text)
        return texts


    def is_trash(self, text):
        """
        Check if a string is trash based on heuristics:

        Args:
            text (str): a string.

        Returns:
            bool: True if the string is trash, False otherwise.
        """

        if len(text) < 5:
            return True
        if sum(char.isupper() for char in text) > len(text)/5:
            return True
        if sum(char.isdigit() for char in text) > len(text)/4:
            return True
        if sum(len(char.encode()) > 1 for char in text) > len(text)/5:
            return True
        if '^' in text:
            return True
        if 'http' in text:
            return True
        if '@' in text:
            return True
        lid = self.identifier.classify(text)
        if lid[0] == 1 and lid[0] != self.lang:
            return True
        return False


def main():
    """
    Read the configuration file, initialize a target data cleaner object
    and preprocess the target corpus.
    """

    with open(sys.argv[3]) as configfile:
        config = json.load(configfile)

    data_cleaner = TargetDataCleaner(config)

    num_sents = 0
    num_del = 0

    with open(sys.argv[1]) as infile, open(sys.argv[2], 'w') as outfile:
        print('Prepare target corpus ...')
        for line in infile:
            num_sents += 1
            texts = data_cleaner.preprocess(line.strip())
            if data_cleaner.is_trash(texts[0]):
                num_del +=1
            else:
                for text in texts:
                    outfile.write(text + '\n')
            if num_sents % 1000 == 0:
                print(f'Processed {num_sents} sentences.\r', end='')
    print(f'Processed {num_sents} sentences.')
    print(f'Done. Deleted {num_del} sentences.')


if __name__ == '__main__':
    main()
