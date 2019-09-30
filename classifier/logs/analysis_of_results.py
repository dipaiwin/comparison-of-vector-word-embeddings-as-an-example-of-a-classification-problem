import re

data = dict()
with open('log.txt', 'r', encoding='utf8') as f:
    for line in f:
        print(line)
        if '\t' in line:
            d = line.split('\t')
            model = d[0].split(':')[1]
            if model not in data:
                data[model] = []
            data[model].append(d[3].split(':')[1])
print(data['glove'])
