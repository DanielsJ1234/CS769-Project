import json
def classify_data(file):
    class_data = []
    with open("classification.txt", 'r') as f:
        class_data = [l.strip("\n") for l in f]
    
    # ent_list = []
    # sum_list = []
    # nom_list = []
    # mult_list = []
    # list_list = []
    # amb_list = []
    # rep_list = []
    # inf_list = []
    # error_list = []
    
    class_dict = {"ent":[], "sum":[], "nom":[], "mult":[], "list":[], 
                  "amb":[], "rep":[], "inf":[], "error": []}
    
    if file.endswith(".json"):
        with open(file) as f:
            data = json.load(f)
        for i, c in enumerate(class_data):
            if "ent" in c:
                # entity-related
                class_dict["ent"].append(data["data"][i])
            if "sum" in c:
                # summarization
                class_dict["sum"].append(data["data"][i])
            if "nom" in c:
                # nomenclature
                class_dict["nom"].append(data["data"][i])
            if "mult" in c:
                # multiple questions asked
                class_dict["mult"].append(data["data"][i])
            if "list" in c:
                # listing data
                class_dict["list"].append(data["data"][i])
            if "amb" in c:
                # ambiguous answers
                class_dict["amb"].append(data["data"][i])
            if "rep" in c:
                # repeating incorrect info
                class_dict["rep"].append(data["data"][i])
            if "inf" in c:
                # inferring operations on data
                class_dict["inf"].append(data["data"][i])
            if "error" in c:
                # error in question or answer
                class_dict["error"].append(data["data"][i])
        file_split = file.split("/")[-1].split(".")
        for k, v in class_dict.items():
            out_file = "".join(file_split[0:-1])+"_"+k+"."+file_split[-1]
            print(out_file)
            with open(out_file, 'w+') as f2:
                temp = {"split":data["split"], "version":data["version"], "data":v}
                json.dump(temp, f2)
            
                
    else:
        with open(file, encoding="utf-8") as f:
            data = [l.strip("\n") for l in f]
        for i, c in enumerate(class_data):
            if "ent" in c:
                # entity-related
                class_dict["ent"].append(data[i])
            if "sum" in c:
                # summarization
                class_dict["sum"].append(data[i])
            if "nom" in c:
                # nomenclature
                class_dict["nom"].append(data[i])
            if "mult" in c:
                # multiple questions asked
                class_dict["mult"].append(data[i])
            if "list" in c:
                # listing data
                class_dict["list"].append(data[i])
            if "amb" in c:
                # ambiguous answers
                class_dict["amb"].append(data[i])
            if "rep" in c:
                # repeating incorrect info
                class_dict["rep"].append(data[i])
            if "inf" in c:
                # inferring operations on data
                class_dict["inf"].append(data[i])
            if "error" in c:
                # error in question or answer
                class_dict["error"].append(data[i])
        file_split = file.split("/")[-1].split(".")
        for k, v in class_dict.items():
            out_file = "".join(file_split[0:-1])+"_"+k+"."+file_split[-1]
            print(out_file)
            with open(out_file, 'w+', encoding="utf-8") as f2:
                #temp = {"split":data["split"], "version":data["version"], "data":v}
                [f2.write(val+"\n") for val in v]
    return data