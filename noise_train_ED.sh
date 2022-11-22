#!/bin/bash
for noise in original 0.1 0.2 0.3 0.6 1.0;
do
  folder=noise_${noise}_$(date +%s)

  python examples/entity_disambiguation/train.py --model-dir=/home/is/gabriel-he/pycharm-upload/luke/data/entity_disambiguation/luke_ed_large/ --dataset-dir=/home/is/gabriel-he/pycharm-upload/luke/data/entity_disambiguation/generated_pipeline_addgold/test_train_data_${noise}/ --titles-file=/home/is/gabriel-he/pycharm-upload/luke/data/entity_disambiguation/enwiki_20181220_titles.txt --redirects-file=/home/is/gabriel-he/pycharm-upload/luke/data/entity_disambiguation/enwiki_20181220_redirects.tsv --output-dir=/home/is/gabriel-he/pycharm-upload/luke/models/ed/${folder} --batch-size 8 --device=cuda:0

  python examples/entity_disambiguation/evaluate.py --model-dir=/home/is/gabriel-he/pycharm-upload/luke/data/entity_disambiguation/luke_ed_large/ --dataset-dir=/home/is/gabriel-he/pycharm-upload/luke/data/entity_disambiguation/generated_pipeline_addgold/test_train_data_${noise}/ --titles-file=/home/is/gabriel-he/pycharm-upload/luke/data/entity_disambiguation/enwiki_20181220_titles.txt --redirects-file=/home/is/gabriel-he/pycharm-upload/luke/data/entity_disambiguation/enwiki_20181220_redirects.tsv --inference-mode=global --document-split-mode=per_mention --device=cuda:1 --output-dir=/home/is/gabriel-he/pycharm-upload/luke/models/ed/${folder}
done
