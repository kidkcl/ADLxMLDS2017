import os
import re
import csv
import pandas as pd

# raw_lines = []
tags = []

with open('tags_clean.csv', 'r') as f:
        all_tags = f.readlines()
        for line in all_tags:
            tokens = re.split(r'\t+|\n|,', line)
            tags.append(tokens)

# remove column signs
for i in range(len(tags)):
    tags[i] = list(filter(lambda a: a != '', tags[i]))
    for j in range(len(tags[i])):
        c = tags[i][j].find(':')
        if c == -1: continue
        else:
            tags[i][j] = tags[i][j][:c]

for i in range(len(tags)):
    new_list = []
    s = ''
    new_list.append(i)
    for j in range(1, len(tags[i])):
        if re.match('^(white|blonde|yellow|orange|black|gray|pink|aqua|purple|green|blue|brown|red).*(eye|hair)', tags[i][j]):
        #if tags[i][j].find('eye') != -1 or tags[i][j].find('hair') != -1:
            s += tags[i][j]
            s += ' '
    new_list.append(s)
    tags[i] = new_list

# remove image with no hair or eye tags
print(tags[0])
tags = list(filter(lambda a: a[1] != '' and len(a[1]) < 25, tags))

longest = 0
longest = max([len(x) for x in tags])
print(len(tags))
df = pd.DataFrame(tags)
#del df[0]
df.to_csv('tags.csv', index=False, header=None)
#with open('tags.csv', 'w') as out:
#    writer = csv.writer(out, delimiter=',')
#    for i in range(len(tags)):
#        writer.writerow(tags[i])
