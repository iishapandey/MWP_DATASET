import json
import spacy
from spacy import displacy
from tabulate import tabulate

with open('data/roth/questions.json') as f:
  data = json.load(f)

nlp = spacy.load("en_core_web_sm")

class UnitTable:
    def __init__(self):
        self.table = []
        self.header = ['ID', 'NUMBER', 'ENTITY', 'ENT_POS', 'ENT_IS_RATE']

    def __str__(self):
        return tabulate(self.table, headers=self.header, tablefmt='pretty')
    
    def append(self, row):
        self.table.append(row)
    
    def size(self):
        return len(self.table), len(self.header)

    def getRow(self, row_id):
        return self.table[row_id]
    
    def getColumn(self, col_id):
        ind = None
        if type(col_id) is int:
            ind = col_id
        else:
            ind = self.header.index(col_id)
        col = []
        for row in self.table:
            col.append(row[ind])
        return col
    
    def addQuestion(self, question, rates):
        pos = 0
        doc = nlp(question)
        for token in doc:
            if token.pos_ == "NUM":
                head = token.head
                if not token.is_sent_start and token.nbor(-1).is_currency:
                    head = token.nbor(-1)
                elif token.nbor(1).is_currency:
                    head = token.nbor(1)
                elif (head.pos_ not in ['NOUN', 'PROPN']) and (token.nbor(1).pos_ in ['NOUN', 'PROPN']):
                    head = token.nbor(1)

                rate = False
                if pos in rates:
                    rate = True

                if head.is_currency:
                    self.table.append(['N' + str(pos), token.text, head.text, 'CURRENCY', rate])
                elif head.tag_ in ['NNS', 'NNPS']:
                    self.table.append(['N' + str(pos), token.text, head.lemma_, head.pos_, rate])
                else:
                    self.table.append(['N' + str(pos), token.text, head.text, head.pos_, rate])
                pos += 1
    
    def updateTable(self):
        status = False
        posList = ['NOUN', 'PROPN', 'CURRENCY']
        posSet = set(posList)

        ets = list(set(self.getColumn('ENTITY')))
        etsPos = set(self.getColumn('ENT_POS'))
        et_inPos = set([row[2] for row in self.table if row[3] in posList])

        # Rule 1 : If all the entity pos are in posList. It is assumed to be correct.
        if len(etsPos - posSet) == 0:
            status = True
        
        # Rule 2 : If none of the entity pos are in posList, then it is wrong.
        elif len(etsPos.intersection(posSet)) == 0:
            status = False

        # Rule 3 : If there is only one entity and not ruled out in Rule 2 . It is assumed to be correct.
        elif len(ets) == 1:
            status = True
        
        # Rule 3 : If there is only one entity with pos in posList, not ruled out by Rule 1,2, then replace all other entities with it.
        elif len(et_inPos) == 1:
            head, pos = None, None
            for row in self.table:
                if row[3] in posList:
                    head, pos = row[2], row[3]
                    break
            for row in self.table:
                if row[3] not in posList:
                    row[2], row[3] = head, pos
            status = True
        
        # Rule 4 : So in all the cases where status is False, replace the entities with pos not in posList replace that enity with 'NK'
        if not status:
            for row in self.table:
                if row[3] not in posList:
                    row[2], row[3] = 'NK', 'NK'
                    
    def getQtEtPair(self, quants, rates):
        QtEtPairs = []
        pos = 0
        if (len(quants) == len(self.table)):
            for i, row in enumerate(self.table):
                rate = False
                if pos in rates:
                    rate = True
                QtEtPairs.append([row[0], quants[i], row[2], rate])
                pos += 1
        else:
            for num in quants:
                rate = False
                if pos in rates:
                    rate = True
                stat = False
                for row in self.table:
                    if row[1] == str(num):
                        QtEtPairs.append(['N' + str(pos), num, row[2], rate])
                        stat = True
                        break
                if not stat:
                    QtEtPairs.append(['N' + str(pos), num, 'NK', rate])
                pos += 1
        return QtEtPairs

    def getTargetUnit(self, question, rates):
        ets = [row[2] for row in self.table if row[3] != 'NK']        
        ets = list(set(ets))
        doc = nlp(question)

        target_num, target_den = 'NK', 'NK'
        start = False
        for token in doc:
            if ((token.text.lower() == 'how') or (token.text.lower() == 'what')):
                start = True
            elif start:
                if token.tag_ in ['NNS', 'NNPS']:
                    if token.lemma_ in ets:
                        target_num = token.lemma_
                        break
                else:
                    if token.text in ets:
                        target_num = token.text
                        break
        start = False
        for token in doc:
            if ((token.text.lower() == 'each') or (token.text.lower() == 'per')):
                start = True
            elif start:
                if token.tag_ in ['NNS', 'NNPS']:
                    if token.lemma_ in ets:
                        target_den = token.lemma_
                        break
                else:
                    if token.text in ets:
                        target_den = token.text
                        break
        if -1 in rates:
            if (target_num != 'NK') and (target_den != 'NK'):
                return [target_num + "/" + target_den, True]
            else:
                return ['NK', True]
        else:
            return [target_num, False]

if __name__ == "__main__":

    for i in range(len(data)):
        table = UnitTable()
        question = data[i]["sQuestion"]
        rates = data[i]["rates"]
        quants = data[i]["quants"]
        table.addQuestion(question, rates)
        table.updateTable()
        data[i]["quantity-units"] = table.getQtEtPair(quants, rates)
        data[i]["target-units"] = table.getTargetUnit(question, rates)
    with open("data/roth/questions_processed.json", "w") as outfile:
        print(json.dumps(data, indent=2), file=outfile)