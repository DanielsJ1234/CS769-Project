import json
import csv

def JSONtoCSV(json_file, csv_dir, file_name):
	with open(json_file) as f:
	  data = json.load(f)

	rows = data['data']
	id = 1

	for row in rows:
		with open(f'{csv_dir}/{file_name}{id}.csv', 'w', newline='') as f:
		    writer = csv.writer(f)
		    tableArr = row['table_array']
		    for tableRow in tableArr:
		    	writer.writerow(tableRow)
		id += 1

	return
	
csv_dir = '../data/csv'
JSONtoCSV('../data/fetaQA-v1_train.json', csv_dir, 'train')
JSONtoCSV('../data/fetaQA-v1_dev.json', csv_dir, 'dev')
JSONtoCSV('../data/fetaQA-v1_test.json', csv_dir, 'test')
