"""
- take single input file with one sentence per line
- remove bad samples from training data
- restrict character range
- tokenize to words
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


class Target_Data_Cleaner():
    def __init__(self, config):
        self.charset = config['charset_target']
        self.lang = config['lang']
        self.n_to_m = config['n_to_m']
        self.n_copies = config['num_copies']
        self.identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        self.identifier.set_languages(config['filter_langs'].append(self.lang))
        self.dic = pyphen.Pyphen(lang='de_DE', left=1, right=1)


    def remove_chars(self, text):
        text = text.replace(r'[_\+\*$#=~\[\]\{\}]', '')
        text = re.sub(r'(- ?)+', '-', text)
        text = re.sub('—', '-', text)
        return text


    def restrict_charset(self, text):
        exceptions = set(c for c in self.charset if len(c.encode()) > 1)
        text = ''.join([unidecode(char) if len(char.encode()) > 1
            and char not in exceptions else char for char in text])
        return text


    def word_tokenize(self, text):
        text = word_tokenize(text)
        return ' '.join(text)


    def _find_nth(self, string, substring, n):
        if (n == 1):
            return string.find(substring)
        else:
            return string.find(substring, self._find_nth(string, substring, n-1) + 1)


    def prepare_n_to_m_mapping(self, text):
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


    def preprocess(self, text, n_to_m=True):
        text = self.remove_chars(text)
        text = self.restrict_charset(text)
        text = self.word_tokenize(text)
        text = re.sub(r' +', ' ', text)
        texts = []
        for i in range(self.n_copies):
            if self.n_to_m:
                texts.append(self.prepare_n_to_m_mapping(text))
            else:
                texts.append(text)
        return texts


    def is_trash(self, text):
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
    with open(sys.argv[3]) as f:
        config = json.load(f)

    data_cleaner = Target_Data_Cleaner(config)

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
