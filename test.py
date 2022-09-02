from allennlp.models.archival import load_archive
from allennlp.common.util import import_module_and_submodules

import_module_and_submodules("examples")
serialization_dir = "studio-ousia/luke-large-finetuned-conll-2003"
archive = load_archive(serialization_dir, cuda_device = -1)

dataset_reader = archive.dataset_reader
model = archive.model

sentence = "Tokyo is the biggest city in Japan ."
# the input must be tokenized into words (not subwords). Be careful of the period in the end.
words = sentence.split()

instances = [instance for instance in dataset_reader.data_to_instance(
    words = words,
    labels = ["O" for _ in range(len(words))], # feed dummy labels
    sentence_boundaries = [0, len(words)],
    doc_index = "foo", # feed a dummy index
)]
# if the sentence is short enough, the instances should only contain one instance.
instance = instances[0]

output_dict = model.forward_on_instance(instance)
non_entity_label_index = model.vocab.get_token_index("O", "labels")

for (s, e), prediction in zip(instance["original_entity_spans"].tensor, output_dict["prediction"]):
    if prediction == non_entity_label_index:
        continue
    entity_label = model.vocab.get_token_from_index(prediction, "labels")
    print(words[s:e],  entity_label)