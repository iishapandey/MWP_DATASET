import json
import spacy
import random
import copy
import constants
from copy import deepcopy
import numpy as np



with open("data/allArith/questions/train.json", 'r') as f:
	train = json.load(f)

with open("data/allArith/questions/test.json", 'r') as f:
	test = json.load(f)

for j in range(len(train)):
	d = train[j]
	ques = d['sQuestion'].split()

	sModques = []

	num=0

	for i in range(len(ques)):
		tok = ques[i]
		try:
			if tok[0]=='$' and len(tok)>1:
				int(tok[1])
				sModques.append("<NUM>")
			else:
				int(tok[0])
				sModques.append("<NUM>")
			num+=1
			
		except ValueError:
			sModques.append(tok)

	train[j]['sModQuestion'] = ' '.join(sModques)

for j in range(len(test)):
	d = test[j]
	ques = d['sQuestion'].split()

	sModques = []

	num=0

	for i in range(len(ques)):
		tok = ques[i]
		try:
			if tok[0]=='$' and len(tok)>1:
				int(tok[1])
				sModques.append("<NUM>")
			else:
				int(tok[0])
				sModques.append("<NUM>")
			num+=1
			
		except ValueError:
			sModques.append(tok)

	test[j]['sModQuestion'] = ' '.join(sModques)




with open("train_final.json", 'w') as f:
	json.dump(train, f, ensure_ascii=False, indent=2)

with open("test_final.json", 'w') as f:
	json.dump(test, f, ensure_ascii=False, indent=2)