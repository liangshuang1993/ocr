from io import open
dicts = ''
with open('/datasets/text_renderer/data/chars/med.txt', encoding='utf-8') as f:
    for ch in f.readlines():
        ch = ch.strip()
        dicts=dicts+ch

char_to_id = {j: i for i,j in enumerate(dicts)}
#print char_to_id

DATASETS = '/datasets/text_renderer/train2'

with open(DATASETS + '/default/tmp_labels.txt', encoding='utf-8') as f:
#with open('/datasets/TextRecognitionDataGenerator/TextRecognitionDataGenerator/train2/labels.txt', encoding='utf-8') as f:
    lines = f.readlines()

import random
random.shuffle(lines)
#with open('/datasets/TextRecognitionDataGenerator/TextRecognitionDataGenerator/train2.txt', 'wb') as f:
with open(DATASETS + '/default/tmp_labels_id.txt', 'wb') as f:
    for line in lines:
        content_list = line.strip().split(' ')
        image = content_list[0]
        label = ''.join(content_list[1:])
        f.write(image + '.jpg')
        for char in label:
            try:
                f.write(' ' + str(char_to_id[char] + 1))
            except Exception as e:
                print char
        f.write('\n')


