import numpy as np
import json
import torch
import re
import string
import argparse
import constants
from copy import deepcopy
from equation_tac import EquationTAC
from data_preprocess import read_data_json, write_data_json
from data_preprocess import Et, expression_list, from_infix_to_postfix, construct_exp_tree, print_exp_tree
import random
from transformers import BertTokenizer, BertModel

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
        question = elem['segIndQuestion'] if use_num_ind else elem['segQuestion']
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
    return vocab2id, id2vocab

# Returns all required features from the data in json
def get_features(data, vocab2id, use_num_ind=False):
    ids, nums, ans, num_features, qn_features, eqns = [], [], [], [], [], []
    num_units, num_rates, ans_units, ans_rates = [], [], [], []
    num_words = 0
    num_unk = 0
    for elem in data:
        id = elem["iIndex"]
        question = elem['segIndQuestion'] if use_num_ind else elem['segQuestion']
        quants = elem['quants']
        quant_units, quant_rates = [], []

        tgt_unit = random.choice(elem['target-units'][0]) if isinstance(elem['target-units'][0], list) else elem['target-units'][0]
        tgt_rate = elem['target-units'][1]

        for unit_info in elem['quantity-units']:
            quant_unit = random.choice(unit_info[2]) if isinstance(unit_info[2], list) else unit_info[2]
            quant_rate = unit_info[3]
            quant_units.append(quant_unit)
            quant_rates.append(quant_rate)

        for num in constants.MATH23K_CONSTANTS:
            if num not in quants:
                quants.append(num)
                quant_units.append('NO')
                quant_rates.append(False)
        
        eqn = elem['lEquations'][0]
        answer = elem['lSolutions'][0]
        words = question.strip().split(' ')

        ids.append(id)
        nums.append(quants)
        ans.append(answer)
        eqns.append(eqn)
        num_units.append(quant_units)
        num_rates.append(quant_rates)
        ans_units.append(tgt_unit)
        ans_rates.append(tgt_rate)

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
    return ids, nums, num_units, num_rates, ans, ans_units, ans_rates, num_features, qn_features, eqns

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
        self._ids, self._nums, self._num_units, self._num_rates, self._ans_list, self._ans_units, self._ans_rates, \
            self._num_features, self._qn_features, self._eqns = get_features(self._data_list, self._vocab2id, self._use_num_ind)

    def save_data(self, path):
        data = {'dict' : 
			{
                'ids': self._ids, 
				'src': self._qn_features, 
				'feat': self._num_features, 
				'nums':self._nums, 
                'num_units':self._num_units, 
                'num_rates':self._num_rates, 
				'ans': self._ans_list, 
                'ans_units':self._ans_units, 
                'ans_rates':self._ans_rates, 
				'vocab' : self._vocab2id, 
				'eqns' : self._eqns
			}
		}
        torch.save(data, path)

# Returns Data class for preprocessing - Bert
class BertData(GruData):
    def __init__(self, path, vocab2id = None, id2vocab = None, use_num_ind = False):
        super().__init__(path, vocab2id, id2vocab, use_num_ind)
        self.max_sequence_length = 200
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.pad_token = 0
        self.sequence_a_segment_id = 0
        self.sequence_b_segment_id = 1
        self.pad_token_segment_id = 0
        self.cls_token_segment_id = 0
        self.mask_padding_with_zero = True
        self.doc_stride = 128
        self.max_query_length = 64
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case = False)
        self._list_input_ids, self._list_token_type_ids, self._list_attention_mask = self.preprocess(path)
	
    def preprocess(self, path):
        data = read_data_json(path)
        list_input_ids, list_token_type_ids, list_attention_mask = [], [], []
        stopwords = ['，', '、', '．', '；', '=']
		
        for row in data:
            text = row["sQuestion"]
            length = len(text)
            for ind in range(length - 1, -1, -1):
                if text[ind] in stopwords:
                    if ((length - ind) < 2): continue
                    context = text[:(ind + 1)]
                    ques = text[(ind + 1):]
                    break
            
            sentence1_tokenized = self.tokenizer.tokenize(context)
            sentence2_tokenized = self.tokenizer.tokenize(ques)

            if len(sentence1_tokenized) + len(sentence2_tokenized) + 3 > self.max_sequence_length:
                continue

            input_ids, token_type_ids, attention_mask = self.encode(sentence1_tokenized, sentence2_tokenized)
            list_input_ids.append(torch.unsqueeze(input_ids,dim=0))
            list_token_type_ids.append(torch.unsqueeze(token_type_ids,dim=0))
            list_attention_mask.append(torch.unsqueeze(attention_mask,dim=0))

        list_input_ids = torch.cat(list_input_ids, dim=0)
        list_token_type_ids = torch.cat(list_token_type_ids, dim=0)
        list_attention_mask = torch.cat(list_attention_mask, dim=0)

        return list_input_ids, list_token_type_ids, list_attention_mask
	
    def encode(self, sentence1, sentence2):
        tokens, segment_mask, input_mask = [], [], []
        tokens.append(self.cls_token)
        segment_mask.append(self.cls_token_segment_id)
        input_mask.append(1 if self.mask_padding_with_zero else 0)

        for tok in sentence1:
            tokens.append(tok)
            segment_mask.append(self.sequence_a_segment_id)
            input_mask.append(1 if self.mask_padding_with_zero else 0)

        tokens.append(self.sep_token)
        segment_mask.append(self.sequence_a_segment_id)
        input_mask.append(1 if self.mask_padding_with_zero else 0)

        for tok in sentence2:
            tokens.append(tok)
            segment_mask.append(self.sequence_b_segment_id)
            input_mask.append(1 if self.mask_padding_with_zero else 0)

        tokens.append(self.sep_token)
        segment_mask.append(self.sequence_b_segment_id)
        input_mask.append(1 if self.mask_padding_with_zero else 0)

        tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        while(len(tokens) < self.max_sequence_length):
            tokens.append(self.pad_token)
            segment_mask.append(self.pad_token_segment_id)
            input_mask.append(0 if self.mask_padding_with_zero else 1)

        tokens = torch.tensor(tokens)
        segment_mask = torch.tensor(segment_mask)
        input_mask = torch.tensor(input_mask)

        return tokens, segment_mask, input_mask
	
    def save_data(self, path):
        data = {'dict' : 
            {
                'input_ids': self._list_input_ids,
                'token_type_ids': self._list_token_type_ids,
                'attention_mask': self._list_attention_mask,
                'ids': self._ids,
                'src': self._qn_features, 
                'feat': self._num_features, 
                'nums':self._nums, 
                'num_units':self._num_units, 
                'num_rates':self._num_rates, 
                'ans': self._ans_list, 
                'ans_units':self._ans_units, 
                'ans_rates':self._ans_rates, 
                'vocab' : self._vocab2id, 
                'eqns' : self._eqns
            }
        }
        torch.save(data, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess code')
    parser.add_argument('-train_json_path', type=str, default='data/math23k/questions/train.json')
    parser.add_argument('-test_json_path', type=str, default='data/math23k/questions/test.json')
    parser.add_argument('-encoder', type=str, default='gru')
    parser.add_argument('-train_save_path', type=str, default='data/math23k/gru/train.pt')
    parser.add_argument('-test_save_path', type=str, default='data/math23k/gru/test.pt')
    parser.add_argument('-min_count', type=int, default=0)
    parser.add_argument('-use_num_ind', action='store_true', default=False)

    args = parser.parse_args()
    train_vocab = get_vocab(read_data_json(args.train_json_path), args.min_count, args.use_num_ind)

    if args.encoder == 'gru':
        train_data = GruData(args.train_json_path, train_vocab[0], train_vocab[1], args.use_num_ind)
        test_data = GruData(args.test_json_path, train_vocab[0], train_vocab[1], args.use_num_ind)
    elif args.encoder == 'bert':
        train_data = BertData(args.train_json_path, args.use_num_ind)
        test_data = BertData(args.test_json_path, train_vocab[0], train_vocab[1], args.use_num_ind)

    train_data.save_data(args.train_save_path)
    test_data.save_data(args.test_save_path)