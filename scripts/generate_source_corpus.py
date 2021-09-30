"""
Generate the source-side corpus based on a target-side corpus
by applying random substring substitutions.
"""

import random
import re
import sys

import json
import pyphen


class SourceDataPreparer():
    """
    Class to generate source-side strings given target-side strings

    Args:
        config (str): path to a JSON configuration file.

    Attributes:
        replacements (dict): dictionary with target-to-source replacements.
        n_to_m (bool): option to activate n-to-m mapping.
        vowels (str): a list of vowels.
        consonants (str): a list of consonants.
    """

    def __init__(self, config):
        self.replacements = config['replacements']
        self.n_to_m = config['n_to_m']
        self.vowels = config['vowels']
        self.consonants = config['consonants']


    def _select(self, lst):
        """
        Randomly select an item from a list.

        Args:
            lst (list): a list with options.

        Returns:
            str: the selected item from the list.
        """

        return lst[random.randint(0, len(lst)-1)]


    def _get_keys(self, position):
        """
        Get the keys per grapheme class in the configuration file (helper function).

        Args:
            position (str): the position in the syllable, e.g. onset or coda.

        Returns:
            str: a '|'-separated string with all keys.
        """

        return '|'.join(sorted(self.replacements[position].keys(), reverse=True))


    def modify(self, grapheme, grapheme_type):
        """
        Apply random substitutions to a grapheme sequence of a specific grapheme type.

        Args:
            grapheme (str): a grapheme sequence reflecting the onset, nucleus, coda etc.
                of a syllable.
            grapheme_type (str): the class of the grapheme sequence.

        Returns:
            A randomly modified version of the input string.
        """

        if random.random() > 0.84 and self.replacements[grapheme_type][grapheme] != []:
            grapheme = self._select(self.replacements[grapheme_type][grapheme])
        return grapheme


    def process(self, text):
        """
        Apply random substitutions to a string.

        Args:
            text (str): the target-side string.

        Returns:
            str: the source-side string.
        """

        if self.n_to_m:
            text = text.replace('_', '')
            text = re.sub(rf'(?=[{self.consonants}{self.vowels}])- ', ' ', text)
        tokens = text.split(' ')
        source = []
        for token_orig in tokens:
            token = f' {token_orig} '
            mod = re.sub(rf'(?<=[ ·])({self._get_keys("onset")})(?=[{self.vowels}])',
                         lambda x: self.modify(x.group(1), 'onset'), token)
            if mod != '':
                token = mod
            mod = re.sub(rf'(?<=[{self.vowels}])({self._get_keys("coda")})(?=[ ·])',
                         lambda x: self.modify(x.group(1), 'coda'), token)
            if mod != '':
                token = mod
            mod = re.sub(rf'(?<=[ ·{self.consonants}])({self._get_keys("nucleus")})',
                         lambda x: self.modify(x.group(1), 'nucleus'), token)
            if mod != '':
                token = mod
            mod = re.sub(rf'(?<=[ ·])([{self.consonants}]?)({self._get_keys("syllable")})(?=[ ·])',
                         lambda x: x.group(1) + self.modify(x.group(2), 'syllable'), token)
            if mod != '':
                token = mod
            mod = re.sub(rf'(?<=[{self.consonants}{self.vowels}·])({self._get_keys("final")})(?= )',
                         lambda x: self.modify(x.group(1), 'final'), token)
            if mod != '':
                token = mod
            mod = re.sub(rf'(?<=[{self.vowels}])'\
                         '({self._get_keys("ambisyllabic")})(?=[{self.vowels} ])',
                         lambda x: self.modify(x.group(1), 'ambisyllabic'), token)
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
    """
    Read the configuration file, initialize a source-data generator
    and generate a source-side corpus by applying random substring substitutions.
    """

    with open(sys.argv[3]) as configfile:
        config = json.load(configfile)

    dic = pyphen.Pyphen(lang=f'{config["lang"]}_{config["lang"].upper()}',
                        left=1, right=1)

    data_preparer = SourceDataPreparer(config)

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
