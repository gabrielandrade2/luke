from allennlp.predictors import Predictor
from allennlp.common.util import import_module_and_submodules

import_module_and_submodules("examples")

predictor = Predictor.from_path(
    "/home/is/gabriel-he/pycharm-upload/luke/results/ner_old/luke-base_noise_0.05_0/model.tar.gz")
results = predictor.predict(sentence="Did Uriah honestly think he could beat The Legend of Zelda in under three hours?")
for word, tag in zip(results["words"], results["tags"]):
    print(f"{word}\t{tag}")
