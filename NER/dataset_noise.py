import os
from argparse import ArgumentParser
from random import random

import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple


def parse_conll_ner_data(input_file: str, encoding: str = "utf-8"):
    words: List[str] = []
    labels: List[str] = []
    p1: List[str] = []
    p2: List[str] = []
    sentence_boundaries: List[int] = [0]

    try:
        with open(input_file, "r", encoding=encoding) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("-DOCSTART"):
                    if words:
                        assert sentence_boundaries[0] == 0
                        assert sentence_boundaries[-1] == len(words)
                        yield words, labels, p1, p2, sentence_boundaries
                        words = []
                        labels = []
                        p1 = []
                        p2 = []
                        sentence_boundaries = [0]
                    continue

                if not line:
                    if len(words) != sentence_boundaries[-1]:
                        sentence_boundaries.append(len(words))
                else:
                    parts = line.split(" ")
                    words.append(parts[0])
                    p1.append(parts[1])
                    p2.append(parts[2])
                    labels.append(parts[3])

        if words:
            yield words, labels, p1, p2, sentence_boundaries
    except UnicodeDecodeError as e:
        raise Exception("The specified encoding seems wrong. Try either ISO-8859-1 or utf-8.") from e


def decision(probability):
    return random() < probability


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--noise_ratio", type=float, required=True)
    args, _ = parser.parse_known_args()

    noise_ratio = args.noise_ratio

    for file in ["train", "testa"]:
        for i in tqdm(range(10)):
            iter = parse_conll_ner_data("data/ner_conll/en/{}.txt".format(file))
            os.makedirs('data/ner_conll/en/noise_{}'.format(noise_ratio), exist_ok=True)
            fout = open('data/ner_conll/en/noise_{}/{}_{}.txt'.format(noise_ratio, file, i), 'w', encoding='utf8')

            for doc in iter:
                labels = doc[1]
                for j in range(len(labels)):
                    if labels[j].startswith('B'):
                        if decision(noise_ratio):
                            # 1 - backward_noise, 2 - forward_noise, 3 - both
                            noise_type = np.random.choice([1, 2, 3])
                            if noise_type & 1:
                                try:
                                    max = 0
                                    edit = False
                                    for k in range(1,2):
                                        if j-k < 0:
                                            break
                                        if labels[j-k].startswith('O'):
                                            max = k
                                        else:
                                            break
                                    for k in range(1, max+1):
                                        if k == max:
                                            labels[j - k] = 'B' + labels[j][1:]
                                        else:
                                            labels[j - k] = 'I' + labels[j][1:]
                                        edit = True
                                    if edit:
                                        labels[j] = 'I' + labels[j][1:]
                                except IndexError:
                                    pass
                            if noise_type & 2:
                                try:
                                    while (labels[j + 1].startswith('I')):
                                        j += 1
                                    for k in range(1,2):
                                        if labels[j + k].startswith('O'):
                                            labels[j + k] = 'I' + labels[j][1:]
                                        else:
                                            break
                                except IndexError:
                                    pass

                fout.write('-DOCSTART- -X- -X- O\n')
                for j in range(len(doc[0])):
                    if j in doc[4]:
                        fout.write('\n')
                    fout.write('{} {} {} {}\n'.format(doc[0][j], doc[2][j], doc[3][j], labels[j]))
                fout.write('\n')

            fout.flush()
            fout.close()
