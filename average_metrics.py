import glob
import json
import os

def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        if key in ['training_f1', 'training_precision', 'training_recall', 'training_strawberry', 'training_span_accuracy', 'training_loss'
                   , 'validation_f1', 'validation_precision', 'validation_recall', 'validation_strawberry', 'validation_span_accuracy', 'validation_loss',
                   'f1', 'precision', 'recall', 'strawberry', 'span_accuracy', 'loss']:
            mean_dict[key] = sum(float(d[key]) for d in dict_list) / len(dict_list)
    return mean_dict

def dict_print(dict):
    for key in dict.keys():
        print('{}\t {}'.format(key, dict[key]))

if __name__ == '__main__':
    for noise in ['original', '0.05', '0.1', '0.2', '0.3', '0.6']:
        if noise == 'original':
            dirs = ['results/ner/luke-base']
        else:
            dirs = glob.glob('results/ner/luke-base_noise_' + noise + '_0')

        training = []
        test = []
        for dir in dirs:
            try:
                training.append(json.loads(open(os.path.join(dir, 'metrics.json')).read()))
                test.append(json.loads(open(os.path.join(dir, 'metrics_test.json')).read()))
            except Exception as e:
               pass

        mean_training = dict_mean(training)
        mean_test = dict_mean(test)

        print('Noise: ' + noise)
        print('Training:')
        dict_print(mean_training)

        print('Test:')
        dict_print(mean_test)
        print('\n')
