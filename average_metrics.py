import glob
import json
import os

import pandas as pd


def dict_mean(dict_list, word, noise):
    mean_dict = {}
    for key in dict_list[0].keys():
        if key in ['training_f1', 'training_precision', 'training_recall', 'training_strawberry', 'training_span_accuracy', 'training_loss'
                   , 'validation_f1', 'validation_precision', 'validation_recall', 'validation_strawberry', 'validation_span_accuracy', 'validation_loss',
                   'f1', 'precision', 'recall', 'strawberry', 'span_accuracy', 'loss']:
            mean_dict[key] = sum(float(d[key]) for d in dict_list) / len(dict_list)
    mean_dict['word'] = word
    mean_dict['noise'] = noise
    return mean_dict

def dict_print(dict):
    for key in dict.keys():
        print('{}\t {}'.format(key, dict[key]))

if __name__ == '__main__':
    t = []
    tt = []
    for word in range(1, 11):
        for noise in ['0.1', '0.2', '0.3', '0.6']:
            dirs = glob.glob('/Users/gabriel-he/Documents/NAIST-PhD/Strawberry score/Luke multiple-word/luke-base_noise_{}_{}_0'.format(word, noise))

            training = []
            test = []
            for dir in dirs:
                try:
                    training.append(json.loads(open(os.path.join(dir, 'metrics.json')).read()))
                    test.append(json.loads(open(os.path.join(dir, 'metrics_test.json')).read()))
                except Exception as e:
                   pass

            mean_training = dict_mean(training, word, noise)
            mean_test = dict_mean(test, word, noise)

            print('Word: ' + str(word))
            print('Noise: ' + noise)
            print('Training:')
            dict_print(mean_training)

            print('Test:')
            dict_print(mean_test)
            print('\n')
            t.append(mean_training)
            tt.append(mean_test)

    df1 = pd.DataFrame(t)
    df2 = pd.DataFrame(tt)

    df1.to_csv('/Users/gabriel-he/Documents/NAIST-PhD/Strawberry score/Luke multiple-word/luke-base_training.csv')
    df2.to_csv('/Users/gabriel-he/Documents/NAIST-PhD/Strawberry score/Luke multiple-word/luke-base_test.csv')
