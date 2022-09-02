import os
from argparse import ArgumentParser
from random import random

import numpy as np
from tqdm import tqdm

def decision(probability):
    return random() < probability

if __name__ == '__main__':

    for noise_ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

        file = open('data/entity_disambiguation/aida_testB.csv')
        os.makedirs('data/entity_disambiguation/noise_' + str(noise_ratio), exist_ok=True)
        for i in tqdm(range(10)):

            fout = open('data/entity_disambiguation/noise_{}/aida_testB_{}.csv'.format(noise_ratio, i), 'w', encoding='utf8')
            for line in file:
                line = line.strip()
                if line == '':
                    fout.write('\n')
                    continue
                line = line.split('\t')
                if decision(noise_ratio):
                    # 1 - backward_noise, 2 - forward_noise, 3 - both
                    noise_type = np.random.choice([1, 2, 3])
                    if noise_type & 1:
                        try:
                            if line[3] != 'EMPTYCTXT':
                                word = line[2]
                                before = line[3].split(' ')
                                line[2] = before[-1] + ' ' + word
                                line[3] = ' '.join(before[:-1])
                        except Exception:
                            pass
                    if noise_type & 2:
                        try:
                            if line[4] != 'EMPTYCTXT':
                                word = line[2]
                                after = line[4].split(' ')
                                line[2] = word + ' ' + after[0]
                                line[4] = ' '.join(after[1:])
                        except Exception:
                            pass
                fout.write('\t'.join(line))
                fout.write('\n')
            fout.close()
