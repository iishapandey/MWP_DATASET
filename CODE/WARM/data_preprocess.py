import numpy as np
import json
import torch
import re
import string
import argparse
import constants
from copy import deepcopy
# from transformers import BertTokenizer, BertModel
from equation_tac import EquationTAC
import random

import spacy 
from spacy.lang.en import English 
nlp = English()
#nlp.add_pipe(nlp.create_pipe('sentencizer'))
nlp.add_pipe('sentencizer')

# Returns data from json
def read_data_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# Writes data into json
def write_data_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dumps(data, indent=2, ensure_ascii=False, file=f)     

# Returns vocab dictionary from the data in json
def get_vocab(data, min_count = 0, use_num_ind=False):
    vocab2id = {'<BOS>':constants.BOS, '<EOS>':constants.EOS}
    id2vocab = {constants.BOS:'<BOS>', constants.EOS:'<EOS>'}
    vocab2count = {'<BOS>':len(data), '<EOS>':len(data)}

    if use_num_ind:
        for i in range(constants.MATH23K_MAXNUM):
            vocab2id["<NUM" + str(i) + ">"] = constants.NUM + i
            id2vocab[constants.NUM + i] = "<NUM" + str(i) + ">"
            vocab2count["<NUM" + str(i) + ">"] = 0
    else:
        vocab2id['<NUM>'] = constants.NUM
        id2vocab[constants.NUM] = '<NUM>'
        vocab2count['<NUM>'] = 0
    
    n_words = constants.NUM + constants.MATH23K_MAXNUM if use_num_ind else constants.NUM + 1
    
    for elem in data:
        question = elem['sIndQuestion'] if use_num_ind else elem['sQuestion']
        words = question.strip().split(' ')
        for word in words:
            if word not in vocab2id.keys():
                vocab2id[word] = n_words
                vocab2count[word]= 1
                id2vocab[n_words] = word
                n_words += 1
            else:
                vocab2count[word] += 1
    
    vocab2id = {'<BOS>':constants.BOS, '<EOS>':constants.EOS}
    id2vocab = {constants.BOS:'<BOS>', constants.EOS:'<EOS>'}

    if use_num_ind:
        for i in range(constants.MATH23K_MAXNUM):
            vocab2id["<NUM" + str(i) + ">"] = constants.NUM + i
            id2vocab[constants.NUM + i] = "<NUM" + str(i) + ">"
    else:
        vocab2id['<NUM>'] = constants.NUM
        id2vocab[constants.NUM] = '<NUM>'
    
    keep_words = []

    for k, v in vocab2count.items():
        if k in vocab2id.keys():
            keep_words.append(k)
        elif v >= min_count:
            keep_words.append(k)

    print('keep_words %s / %s = %.4f' % (
        len(keep_words) + 2, n_words, (len(keep_words) + 2) / n_words
    ))

    n_words = constants.NUM + constants.MATH23K_MAXNUM if use_num_ind else constants.NUM + 1

    for word in keep_words:
        if word not in vocab2id.keys():
            vocab2id[word] = n_words
            id2vocab[n_words] = word
            n_words += 1
    # print("########################")
    # print("vocab2id",vocab2id)
    # print("id2vocab",id2vocab)
    # print("vocab2count",vocab2count)      
    # print("########################")  
    return vocab2id, id2vocab

# An expression tree node
class Et:
    # Constructor to create a node
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    # Prints the expression tree given its root
    # def printEt(self, root, space=0):
    #     space += 10
    #     if root.right is not None:
    #         self.printEt(root.right, space)
    #         print()
    #     for _ in range(10, space):
    #         print(end=" ")
    #     print(root.value)
    #     if root.left is not None:
    #        self. printEt(root.left, space)

    # Prints the equation given the expression tree
    def printeqn(self, root):
        if root is None:
            return ''
        else:
            left = self.printeqn(root.left)
            right = self.printeqn(root.right)
            val = str(root.value) if root.value is not None else ''
            if left != '':
                left = '( ' + left
            if right != '':
                right = right + ' )'
            ans = left + ' ' + val + ' ' + right
            #print("peqn", left, val, right)
            return ' '.join(ans.split())

    def print_eqn(self, root):
        return self.printeqn(root)

# Returns expression from expression list
def expression_list(expression):
    res = list()
    st = ''
    for e in expression:
        if e in "0123456789.":
            st += e
        else:
            if st != '':
                res.append(st)
                st = ''
            res.append(e)
    if st != '':
        res.append(st)
        st = ''
    return res

# Returns postfix list of the infix expression
def from_infix_to_postfix(expression):
    exp_list = expression_list(expression)
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    for e in exp_list:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in ["(", "["] and priority[e] <= priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    return res

# Returns root of constructed tree for given equation
def construct_exp_tree(equation):
    expression = equation[2:]
    postfix = from_infix_to_postfix(expression)
    stack = []

    # Traverse through every character of input expression
    for char in postfix:

        # if operand, simply push into stack
        if char not in ["+", "-", "*", "/", "^"]:
            t = Et(char)
            stack.append(t)
        # Operator
        else:
            # Pop two top nodes
            t = Et(char)
            t1 = stack.pop()
            t2 = stack.pop()

            # make them children
            t.right = t1
            t.left = t2

            # Add this subexpression to stack
            stack.append(t)
    # Only element  will be the root of expression tree
    t = stack.pop()
    return t

# Prints the expression tree given the equation
def print_exp_tree(equation):
    exp_tree = construct_exp_tree(equation)
    exp_tree.printEt()

# Returns all required features from the data in json
def get_features(data, vocab2id, use_num_ind=False):
    ids, nums, ans, num_features, qn_features, eqns = [], [], [], [], [], []
    #num_units, num_rates, ans_units, ans_rates = [], [], [], []
    num_words = 0
    num_unk = 0
    for elem in data:
        id = elem["iIndex"]
        question = elem['sIndQuestion'] if use_num_ind else elem['sQuestion']
        quants = elem['quants']
       # quant_units, quant_rates = [], []

        #tgt_unit = random.choice(elem['target-units'][0]) if isinstance(elem['target-units'][0], list) else elem['target-units'][0]
        #tgt_rate = elem['target-units'][1]

       # for unit_info in elem['quantity-units']:
       #     quant_unit = random.choice(unit_info[2]) if isinstance(unit_info[2], list) else unit_info[2]
       #     quant_rate = unit_info[3]
       #     quant_units.append(quant_unit)
       #     quant_rates.append(quant_rate)

        #uncomment for math23k
        # for num in constants.MATH23K_CONSTANTS:
        #     if num not in quants:
        #         quants.append(num)
        #         quant_units.append('NO')
        #         quant_rates.append(False)
        
        eqn = elem['lEquations'][0]
        answer = elem['lSolutions'][0]
        words = question.strip().split(' ')

        ids.append(id)
        nums.append(quants)
        ans.append(answer)
        eqns.append(eqn)
        #num_units.append(quant_units)
        #num_rates.append(quant_rates)
        #ans_units.append(tgt_unit)
        #ans_rates.append(tgt_rate)

        qn_feature = [vocab2id['<BOS>']]
        qn_num_feature = [0]
        for word in words:
            num_words += 1
            if word[:4] != '<NUM':
                qn_num_feature.append(0)
                if word not in vocab2id.keys():
                    num_unk += 1
                    qn_feature.append(constants.UNK)
                else:
                    qn_feature.append(vocab2id[word])
            else:
                qn_feature.append(vocab2id[word])
                qn_num_feature.append(1)
        qn_num_feature.append(0)
        qn_feature.append(vocab2id['<EOS>'])
        num_features.append(qn_num_feature)
        qn_features.append(qn_feature)
    print('fraction of unks %s / %s = %.4f' % (
        num_unk, num_words, num_unk / num_words
    ))
    #return ids, nums, num_units, num_rates, ans, ans_units, ans_rates, num_features, qn_features, eqns
    return ids, nums, ans, num_features, qn_features, eqns

# Returns Data class for preprocessing - GRU
class GruData:
    def __init__(self, path, vocab2id = None, id2vocab = None, use_num_ind = False):
        filename = path
        self._data_list = read_data_json(filename)
        self._use_num_ind = use_num_ind
        if (vocab2id is None) or (id2vocab is None):
            self._vocab2id, self._id2vocab = get_vocab(self._data_list)
        else:
            self._vocab2id = vocab2id
            self._id2vocab = id2vocab
        #self._ids, self._nums, self._num_units, self._num_rates, self._ans_list, self._ans_units, self._ans_rates, \
         #   self._num_features, self._qn_features, self._eqns = get_features(self._data_list, self._vocab2id, self._use_num_ind)

            self._ids, self._nums, self._ans_list, self._num_features, self._qn_features, self._eqns = get_features(self._data_list, self._vocab2id, self._use_num_ind)


    def save_data(self, path):
        data = {'dict' : 
            {
                'ids': self._ids, 
                'src': self._qn_features, 
                'feat': self._num_features, 
                'nums':self._nums, 
                #'num_units':self._num_units, 
                #'num_rates':self._num_rates, 
                'ans': self._ans_list, 
                #'ans_units':self._ans_units, 
                #'ans_rates':self._ans_rates, 
                'vocab' : self._vocab2id, 
                'eqns' : self._eqns
            }
        }
        
        torch.save(data, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess code')
    parser.add_argument('-train_json_path', type=str, default='data/ixl/questions/train.json')
    parser.add_argument('-test_json_path', type=str, default='data/ixl/questions/test.json')
    parser.add_argument('-encoder', type=str, default='gru')
    parser.add_argument('-train_save_path', type=str, default='DATA/train.pt')
    parser.add_argument('-test_save_path', type=str, default='DATA/test.pt')
    parser.add_argument('-min_count', type=int, default=0)
    parser.add_argument('-use_num_ind', action='store_true', default=False)

    args = parser.parse_args()
    train_vocab = get_vocab(read_data_json(args.train_json_path), args.min_count, args.use_num_ind)

    if args.encoder == 'gru':
        train_data = GruData(args.train_json_path, train_vocab[0], train_vocab[1], args.use_num_ind)
        test_data = GruData(args.test_json_path, train_vocab[0], train_vocab[1], args.use_num_ind)
 

    train_data.save_data(args.train_save_path)
    test_data.save_data(args.test_save_path)
