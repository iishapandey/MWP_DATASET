import json
from os import replace
import re
import argparse


parser = argparse.ArgumentParser(description='preprocess code')
parser.add_argument('-mode', type=str, default='train')
parser.add_argument('-json_path', type=str, default='hindi_train_processed.json')
#parser.add_argument('-test_json_path', type=str, default='english_test.json')
args = parser.parse_args()

f = open(args.json_path)
data = json.load(f) 

out = []

for mwp in data:
    sModQuestion = re.sub(r" [0-9]+ | [0-9]+.[0-9]+ | [0-9]+.[0-9]+. | *[0-9]|+.[0-9]*", " <NUM> ", mwp["original_text"])
    out_dict = {
        "iIndex" : mwp["id"],
        "sQuestion" : mwp['original_text'],
        "sModQuestion" : sModQuestion,
        "sIndQuestion" : mwp['text'],
        "quants" : mwp['num_list'],
        "lEquations" : [["X=?"]],
        "lSolutions" : [mwp['ans']]
    }
    out.append(out_dict)
with open(args.json_path, "w") as f:
    json_obj = json.dumps(out, indent = 5, ensure_ascii = False)
    f.write(json_obj)
