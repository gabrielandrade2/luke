#!/bin/bash
for words in 6 7 8 9 10;
do
for noise in 0.1 0.2 0.3 0.6 1.0;
do
for i in $(eval echo "0");
do
    export TRANSFORMERS_MODEL_NAME="studio-ousia/luke-base";
    export TRAIN_DATA_PATH="data/new_data/en/noise_${words}_${noise}/train_${i}.txt";
    export VALIDATION_DATA_PATH="data/new_data/en/noise_${words}_${noise}/testa_${i}.txt";

    if [ -d "results/ner/luke-base_noise_${words}_${noise}_${i}" ]
    then
      allennlp train examples/ner/configs/transformers_luke_with_entity_aware_attention.jsonnet -s results/ner/luke-base_noise_${words}_${noise}_${i} --include-package examples -o '{"trainer.cuda_device": '$1', "trainer.use_amp": true' --recover
    else
      allennlp train examples/ner/configs/transformers_luke_with_entity_aware_attention.jsonnet -s results/ner/luke-base_noise_${words}_${noise}_${i} --include-package examples -o '{"trainer.cuda_device": '$1', "trainer.use_amp": true}'
    fi

    allennlp evaluate results/ner/luke-base_noise_${words}_${noise}_${i} data/ner_conll/en/testb.txt --include-package examples --output-file results/ner/luke-base_noise_${words}_${noise}_${i}/metrics_test.json --cuda $1
done
done
done
