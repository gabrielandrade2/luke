#!/bin/bash

for i in $(eval echo "{$3..3}");
do
    export TRANSFORMERS_MODEL_NAME="studio-ousia/luke-base";
    export TRAIN_DATA_PATH="data/ner_conll/en/noise_${1}/train_${i}.txt";
    export VALIDATION_DATA_PATH="data/ner_conll/en/noise_${1}/testa_${i}.txt";

    if [ -d "results/ner/luke-base_100_noise_${1}_${i}" ]
    then
      allennlp train examples/ner/configs/transformers.jsonnet -s results/ner/luke-base_100_noise_${1}_${i} --include-package examples -o '{"trainer.cuda_device": '$2', "trainer.use_amp": true, "trainer.num_epochs": 100}' --recover
    else
      allennlp train examples/ner/configs/transformers.jsonnet -s results/ner/luke-base_100_noise_${1}_${i} --include-package examples -o '{"trainer.cuda_device": '$2', "trainer.use_amp": true, "trainer.num_epochs": 100}'
    fi

    allennlp evaluate results/ner/luke-base_100_noise_${1}_${i} data/ner_conll/en/testb.txt --include-package examples --output-file results/ner/luke-base_100_noise_${1}_${i}/metrics_test.json --cuda $2
done
