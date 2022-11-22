python examples/entity_disambiguation/evaluate.pyimport glob
import json
import os

import pandas as pd

output = {}
base_folder = 'results/ed_noise_miss_nochange'
folders = sorted(os.listdir(base_folder))
for folder in folders:
    files = glob.glob(base_folder + "/" + folder + "/eval_results_*.txt")
    for file in files:
        with open(file, 'r') as f:
            data = json.loads(f.read())
            data = data['test_b']
            output[folder] = data

df = pd.DataFrame(output).transpose()