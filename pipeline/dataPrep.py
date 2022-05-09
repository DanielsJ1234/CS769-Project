import json
import csv

with open('../data/fetaQA-v1_train.json') as f:
  data = json.load(f)

#print(json.dumps(data, indent = 4, sort_keys=True))

rows = data['data']
id = 1

with open('trainForTapas.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'question', 'table_file', 'answer_coordinates', 'answer_text', 'aggregation_label', 'float_answer'])
    for row in rows:
        writer.writerow([id, row['question'], f'../data/csv/train{i}.csv'])
        id += 1

# {
#     "answer": "Straumen Chapel is a church in the S\u00f8rreisa parish which is part of the Senja prosti (deanery).",
#     "feta_id": 21164,
#     "highlighted_cell_ids": [
#         [
#             14,
#             1
#         ],
#         [
#             15,
#             1
#         ],
#         [
#             16,
#             1
#         ],
#         [
#             16,
#             2
#         ]
#     ],
#     "page_wikipedia_url": "http://en.wikipedia.org/wiki/List_of_churches_in_Troms",
#     "question": "Straumen Chapel is part of which Parish ?",
#     "source": "mturk-not-evaluated",
#     "table_array": [
#         [
#             "Municipality",
#             "Parish (sokn)",
#             "Church",
#             "Location",
#             "Year built",
#             "Photo"
#         ],
#         [
#             "Berg",
#             "Berg",
#             "Berg Church",
#             "Skaland",
#             "1955",
#             "Berg kirke, Skaland.JPG"
#         ],
#         [
#             "Berg",
#             "Berg",
#             "Finns\u00e6ter Chapel",
#             "Finns\u00e6ter",
#             "1982",
#             "-"
#         ],
#         [
#             "Berg",
#             "Berg",
#             "Mefjordv\u00e6r Chapel",
#             "Mefjordv\u00e6r",
#             "1916",
#             "-"
#         ],
#         [
#             "Dyr\u00f8y",
#             "Dyr\u00f8y",
#             "Dyr\u00f8y Church",
#             "Holm",
#             "1880",
#             "-"
#         ],
#         [
#             "Dyr\u00f8y",
#             "Dyr\u00f8y",
#             "Br\u00f8stad Chapel",
#             "Br\u00f8stadbotn",
#             "1937",
#             "-"
#         ],
#         [
#             "Lenvik",
#             "Lenvik",
#             "Finnsnes Church",
#             "Finnsnes",
#             "1979",
#             "-"
#         ],
#         [
#             "Lenvik",
#             "Lenvik",
#             "Lenvik Church",
#             "Bjorelvnes",
#             "1879",
#             "-"
#         ],
#         [
#             "Lenvik",
#             "Lenvik",
#             "Rossfjord Church",
#             "Rossfjordstraumen",
#             "1822",
#             "-"
#         ],
#         [
#             "Lenvik",
#             "Lenvik",
#             "Fjordg\u00e5rd Chapel",
#             "Fjordg\u00e5rd",
#             "1976",
#             "-"
#         ],
#         [
#             "Lenvik",
#             "Lenvik",
#             "Gibostad Chapel",
#             "Gibostad",
#             "1939",
#             "-"
#         ],
#         [
#             "Lenvik",
#             "Lenvik",
#             "Hus\u00f8y Chapel",
#             "Hus\u00f8y i Senja",
#             "1957",
#             "-"
#         ],
#         [
#             "Lenvik",
#             "Lenvik",
#             "Lysbotn Chapel",
#             "Lysnes",
#             "1970",
#             "-"
#         ],
#         [
#             "Lenvik",
#             "Lenvik",
#             "Sandbakken Chapel",
#             "Sandbakken",
#             "1974",
#             "-"
#         ],
#         [
#             "S\u00f8rreisa",
#             "S\u00f8rreisa",
#             "S\u00f8rreisa Church",
#             "T\u00f8mmervika",
#             "1992",
#             "-"
#         ],
#         [
#             "S\u00f8rreisa",
#             "S\u00f8rreisa",
#             "Sk\u00f8elv Chapel",
#             "Sk\u00f8elva",
#             "1966",
#             "-"
#         ],
#         [
#             "S\u00f8rreisa",
#             "S\u00f8rreisa",
#             "Straumen Chapel",
#             "1973",
#             "-",
#             "-"
#         ],
#         [
#             "Torsken",
#             "Torsken",
#             "Torsken Church",
#             "Torsken",
#             "1784",
#             "-"
#         ],
#         [
#             "Torsken",
#             "Torsken",
#             "Flakkstadv\u00e5g Chapel",
#             "Flakstadv\u00e5g",
#             "1925",
#             "-"
#         ],
#         [
#             "Torsken",
#             "Torsken",
#             "Gryllefjord Chapel",
#             "Gryllefjord",
#             "1902",
#             "Gryllefjord kapell.JPG"
#         ],
#         [
#             "Torsken",
#             "Torsken",
#             "Medby Chapel",
#             "Medby",
#             "1890",
#             "-"
#         ],
#         [
#             "Tran\u00f8y",
#             "Tran\u00f8y",
#             "Stonglandet Church",
#             "Stonglandseidet",
#             "1896",
#             "-"
#         ],
#         [
#             "Tran\u00f8y",
#             "Tran\u00f8y",
#             "Tran\u00f8y Church",
#             "Tran\u00f8ya",
#             "1775",
#             "-"
#         ],
#         [
#             "Tran\u00f8y",
#             "Tran\u00f8y",
#             "Skrolsvik Chapel",
#             "Skrollsvika",
#             "1924",
#             "-"
#         ],
#         [
#             "Tran\u00f8y",
#             "Tran\u00f8y",
#             "Vangsvik Chapel",
#             "Vangsvik",
#             "1975",
#             "-"
#         ]
#     ],
#     "table_page_title": "List of churches in Troms",
#     "table_section_title": "Senja prosti",
#     "table_source_json": "totto_source/train_json/example-13463.json"
# },
