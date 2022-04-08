import json
import csv

#read data
f = open('data/fetaQA-v1_test.json', 'r')
data = json.load(f)

#read predictions
p = open('outputs/test_preds_seq2seq_base_10.txt', 'r')
lines = p.readlines()


#write to csv file
c = open('data/comparison.csv', 'w', encoding='UTF8', newline='')
writer = csv.writer(c)	
#write header
header = ['page_wikipedia_url', 'question', 'answer', 'prediction']
writer.writerow(header)

i = 0
for d in data['data']:
	arr = []
	arr.append(d['page_wikipedia_url'])
	arr.append(d['question'])
	arr.append(d['answer'])
	arr.append(lines[i].strip())
	writer.writerow(arr)
	i+=1

f.close()
p.close()
c.close()