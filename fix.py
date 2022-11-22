file = open('/home/is/gabriel-he/pycharm-upload/luke/data/entity_disambiguation/generated_pipeline/test_train_data_original/testa_testb_aggregate_original')
fileout = open('/home/is/gabriel-he/pycharm-upload/luke/data/entity_disambiguation/generated_pipeline/test_train_data_original/testa_testb_aggregate_original_fix', 'w')
lines = file.readlines()

a = '1163testb'
b = None
temp = None

for line in lines:
    if line.startswith('-DOCSTART-'):
        temp = line[line.find("(")+1:line.find(")")].split(' ')[0]
        word = line[line.find("(")+1:line.find(")")].split(' ')[1]
        line = '-DOCSTART- (' + a + ' ' + word + ')\n'
        a = temp
    fileout.write(line)


file.close()
fileout.close()

