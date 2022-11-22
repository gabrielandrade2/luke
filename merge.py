import copy

if __name__ == "__main__":

    with open('/Users/gabriel-he/PycharmProjects/luke/data/entity_disambiguation/aida_testB.csv', 'r') as f:
        original = f.readlines()
        originalc = copy.deepcopy(original)
    file = '1.0'
    with open('/Users/gabriel-he/PycharmProjects/luke/data/entity_disambiguation/generated_pipeline_new/test_train_data_{}/aida_testB.csv'.format(file), 'r') as f:
        predicted = f.readlines()

    predicted_unique = []
    for line in predicted:
        match = False
        parsed = line.split('\t')
        for l in original:
            parsed_original = l.split('\t')
            if parsed[0] == parsed_original[0] and parsed[1] == parsed_original[1] and parsed[2] == parsed_original[2]:
                original.remove(l)
                match = True
                break
            elif parsed[0] == parsed_original[0] and parsed[1] == parsed_original[1] and (parsed[2] in parsed_original[2]):
                predicted_unique.append(line)
                original.remove(l)
                originalc.remove(l)
                match = True
                break
        if not match:
            predicted_unique.append(line)

    with open('/Users/gabriel-he/PycharmProjects/luke/data/entity_disambiguation/generated_pipeline_new_addgold/test_train_data_{}/aida_testB.csv'.format(file), 'w') as f:
        f.writelines(predicted_unique)
        f.writelines(originalc)
