import random
import re
import sys

import json
import pyphen
from unidecode import unidecode


class Source_Data_Preparer():
    def __init__(self, config):
        self.replacements = config['replacements']
        self.n_to_m = config['n_to_m']
        self.vowels = config['vowels']
        self.consonants = config['consonants']


    def _select(self, lst):
        return lst[random.randint(0, len(lst)-1)]


    def _get_keys(self, position):
        return '|'.join(sorted(self.replacements[position].keys(), reverse=True))


    def modify(self, grapheme, grapheme_type):
        if random.random() > 0.84 and self.replacements[grapheme_type][grapheme] != []:
            grapheme = self._select(self.replacements[grapheme_type][grapheme])
        return grapheme


    def process(self, text):
        if self.n_to_m:
            text = text.replace('_', '')
            text = re.sub(rf'(?=[{self.consonants}{self.vowels}])- ', ' ', text)
        tokens = text.split(' ')
        source = []
        for token_orig in tokens:
            token = f' {token_orig} '
            mod = re.sub(rf'(?<=[ ·])({self._get_keys("onset")})(?=[{self.vowels}])', lambda x: self.modify(x.group(1), 'onset'), token)
            if mod != '':
                token = mod
            mod = re.sub(rf'(?<=[{self.vowels}])({self._get_keys("coda")})(?=[ ·])', lambda x: self.modify(x.group(1), 'coda'), token)
            if mod != '':
                token = mod
            mod = re.sub(rf'(?<=[ ·{self.consonants}])({self._get_keys("nucleus")})', lambda x: self.modify(x.group(1), 'nucleus'), token)
            if mod != '':
                token = mod
            mod = re.sub(rf'(?<=[ ·])([{self.consonants}]?)({self._get_keys("syllable")})(?=[ ·])', lambda x: x.group(1) + self.modify(x.group(2), 'syllable'), token)
            if mod != '':
                token = mod
            mod = re.sub(rf'(?<=[{self.consonants}{self.vowels}·])({self._get_keys("final")})(?= )', lambda x: self.modify(x.group(1), 'final'), token)
            if mod != '':
                token = mod
            mod = re.sub(rf'(?<=[{self.vowels}])({self._get_keys("ambisyllabic")})(?=[{self.vowels} ])', lambda x: self.modify(x.group(1), 'ambisyllabic'), token)
            if mod != '':
                token = mod
            token = re.sub(r'·', '', token)
            token = re.sub(r'(.)\1+', r'\1\1', token)
            if random.random() > 0.5:
                token = re.sub(r'(.)\1(.)\2', r'\1\2\2', token)
            token = token.lower().strip()
            if token != '':
                source.append(token)
            else:
                source.append(token_orig)
        return re.sub(r' +', ' ', ' '.join(source))


def main():
    with open(sys.argv[3]) as f:
        config = json.load(f)

    dic = pyphen.Pyphen(lang=f'{config["lang"]}_{config["lang"].upper()}', left=1, right=1)

    data_preparer = Source_Data_Preparer(config)

    num_sents = 0

    with open(sys.argv[1]) as infile, open(sys.argv[2], 'w') as outfile:
        print('Prepare source corpus ...')
        for line in infile:
            num_sents += 1
            segmented = []
            tokens = line.strip().split()
            for token in tokens:
                segmented.append(dic.inserted(token, hyphen='·'))
            line = ' '.join(segmented)
            line = data_preparer.process(line.lower().strip())
            outfile.write(line + '\n')
            if num_sents % 1000 == 0:
                print(f'Processed {num_sents} sentences.\r', end='')
    print(f'Processed {num_sents} sentences.')
    print('Done.')

if __name__ == '__main__':
    main()
