import numpy as np
import torch
import torch.utils.data
import constants
from unit_kb import Unit

'''
def collate_fn_gru(insts):
    ids, src_insts, src_feat_insts, num_insts, num_unit_insts, num_rate_insts, ans_insts,\
        ans_unit_insts, ans_rate_insts, eqns_insts = list(zip(*insts))
    src_insts, src_feat_insts, src_lengths = pad_insts(src_insts, src_feat_insts)
    num_insts, num_unit_insts, num_rate_insts, num_lengths, num_masks = pad_num_vocab(num_insts, num_unit_insts, num_rate_insts)
    ans_insts, ans_unit_insts = pad_ans_vocab(ans_insts, ans_unit_insts)
    return (ids, src_insts, src_feat_insts, src_lengths, num_insts, num_unit_insts, num_rate_insts, num_lengths, num_masks, ans_insts, ans_unit_insts, ans_rate_insts, eqns_insts)

def collate_fn_bert(insts):
    input_ids_insts, token_type_ids_insts, attn_mask_insts, ids, src_insts, src_feat_insts, num_insts, num_unit_insts, num_rate_insts, \
        ans_insts, ans_unit_insts, ans_rate_insts, eqns_insts = list(zip(*insts))
    src_insts, src_feat_insts, src_lengths = pad_insts(src_insts, src_feat_insts)
    num_insts, num_unit_insts, num_rate_insts, num_lengths, num_masks = pad_num_vocab(num_insts, num_unit_insts, num_rate_insts)
    ans_insts, ans_unit_insts = pad_ans_vocab(ans_insts, ans_unit_insts)
    input_ids_insts = torch.stack(input_ids_insts, dim=0)
    token_type_ids_insts = torch.stack(token_type_ids_insts, dim=0)
    attn_mask_insts = torch.stack(attn_mask_insts, dim=0)
    return (input_ids_insts, token_type_ids_insts, attn_mask_insts, ids, src_insts, src_feat_insts, src_lengths, num_insts, num_unit_insts, num_rate_insts, num_lengths, num_masks, ans_insts, ans_unit_insts, ans_rate_insts, eqns_insts)
'''

#padding and aligning the dataset
def collate_fn_gru(insts):
            ids, src_insts, src_feat_insts, num_insts, ans_insts, eqns_insts = list(zip(*insts))
            src_insts, src_feat_insts, src_lengths = pad_insts(src_insts, src_feat_insts)
            num_insts, num_lengths, num_masks = pad_num_vocab(num_insts)
            ans_insts = pad_ans_vocab(ans_insts)
            return (ids, src_insts, src_feat_insts, src_lengths, num_insts, num_lengths, num_masks, ans_insts, eqns_insts)
                                    

# def collate_fn_bert(insts):
#             input_ids_insts, token_type_ids_insts, attn_mask_insts, ids, src_insts, src_feat_insts, num_insts, \
#             ans_insts, eqns_insts = list(zip(*insts))
#             src_insts, src_feat_insts, src_lengths = pad_insts(src_insts, src_feat_insts)
#             num_insts, num_lengths, num_masks = pad_num_vocab(num_insts)
#             ans_insts = pad_ans_vocab(ans_insts)
#             input_ids_insts = torch.stack(input_ids_insts, dim=0)
#             token_type_ids_insts = torch.stack(token_type_ids_insts, dim=0)
#             attn_mask_insts = torch.stack(attn_mask_insts, dim=0)
#             return (input_ids_insts, token_type_ids_insts, attn_mask_insts, ids, src_insts, src_feat_insts, src_lengths, num_insts, num_lengths, num_masks, ans_insts, eqns_insts)


def pad_num_vocab(nums):
    batch_size = len(nums)
    num_lengths = [len(num) for num in nums]
    max_seq_len = max(num_lengths)

    num_insts = np.zeros((batch_size, max_seq_len))
    #num_unit_insts = np.zeros((batch_size, max_seq_len), dtype=Unit)
    #num_rate_insts = np.zeros((batch_size, max_seq_len), dtype=bool)
    num_masks = np.zeros((batch_size, max_seq_len))
    for i in range(batch_size):
        for j in range(max_seq_len):
            try:
                num_insts[i, j] = nums[i][j]
               # num_unit_insts[i, j] = Unit(num_units[i][j], nums[i][j])
               # num_rate_insts[i, j] = num_rates[i][j]
                num_masks[i, j] = 1
            except IndexError:
                num_insts[i, j] = 0
               #num_unit_insts[i, j] = Unit('NO')
               # num_rate_insts[i, j] = False

    num_insts = torch.tensor(num_insts)
    num_lengths = torch.tensor(num_lengths)
    num_masks = torch.tensor(num_masks)

    #return num_insts, num_unit_insts, num_rate_insts, num_lengths, num_masks
    return num_insts, num_lengths, num_masks

def pad_ans_vocab(ans_insts):
    batch_size = len(ans_insts)
    #ans_units = np.zeros(batch_size, dtype=Unit)
    #for i in range(batch_size):
       # ans_units[i] = Unit(ans_unit_insts[i], ans_insts[i])
    #return ans_insts, tuple(ans_units)
    return ans_insts

def pad_insts(src, feat):
    batch_size = len(src)
    src_lengths = [len(ques) for ques in src]
    max_seq_len = max(src_lengths)

    src_insts = np.zeros((max_seq_len, batch_size), dtype=np.int64)
    src_feat_insts = np.zeros((max_seq_len, batch_size), dtype=np.int64)

    for i in range(max_seq_len):
        for j in range(batch_size):
            try:
                src_insts[i, j] = src[j][i]
                src_feat_insts[i, j] = feat[j][i]
            except IndexError:
                src_insts[i, j] = constants.PAD
                src_feat_insts[i, j] = constants.PAD

    src_insts = torch.LongTensor(src_insts)
    src_feat_insts = torch.LongTensor(src_feat_insts)
    src_lengths = torch.LongTensor(src_lengths)

    return src_insts, src_feat_insts, src_lengths


class MWP_Gru_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, ids, word2idx, feat2idx,
                src_insts=None, src_feat_insts=None, num_insts=None, ans=None, eqns=None):
        assert src_insts and num_insts
        assert len(src_insts) == len(src_feat_insts)
        self._ids = ids
        self._word2idx = word2idx
        self._feat2idx = feat2idx
        self._idx2word = {idx: word for word, idx in word2idx.items()}
        self._src_insts = src_insts
        self._src_feat_insts = src_feat_insts
        self._num_insts = num_insts
        self._ans = ans
        self._eqns = eqns

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._word2idx

    @property
    def feat2idx(self):
        '''property features2idx'''
        return self._feat2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._idx2word

    def __len__(self):
        return self.n_insts


    def __getitem__(self, idx):
        if self._ans is not None:
            return self._ids[idx], self._src_insts[idx], self._src_feat_insts[idx], self._num_insts[idx], self._ans[idx], self._eqns[idx]
        return self._ids[idx], self._src_insts[idx], self._src_feat_insts[idx], self._num_insts[idx], self._ans[idx], None, None


'''
class MWP_Bert_Dataset(MWP_Gru_Dataset):
    def __init__(self, input_ids, token_type_ids, attention_mask, ids, word2idx, feat2idx, 
                 src_insts=None, src_feat_insts=None, num_insts=None, num_unit_insts=None,
                 num_rate_insts=None, ans=None, ans_units=None, ans_rates=None, eqns=None):
        super().__init__(word2idx, feat2idx, ids, src_insts, src_feat_insts, num_insts, num_unit_insts,\
            num_rate_insts,ans, ans_units, ans_rates, eqns)
        self._input_ids = input_ids
        self._token_type_ids = token_type_ids
        self._attention_mask = attention_mask
	
    def __len__(self):
        return self._input_ids.shape[0]
    
    def __getitem__(self, idx):
        if self._ans is not None:
            return self._input_ids[idx], self._token_type_ids[idx], self._attention_mask[idx], self._ids[idx], self._src_insts[idx], self._src_feat_insts[idx],\
                 self._num_insts[idx], self._num_unit_insts[idx], self._num_rate_insts[idx], self._ans[idx], self._ans_units[idx], \
                     self._ans_rates[idx], self._eqns[idx]
        return self._input_ids[idx], self._token_type_ids[idx], self._attention_mask[idx], self._ids[idx], self._src_insts[idx], self._src_feat_insts[idx],\
                 self._num_insts[idx], self._num_unit_insts[idx], self._num_rate_insts[idx], self._ans[idx], self._ans_units[idx], \
                     None, None

'''
# class MWP_Bert_Dataset(MWP_Gru_Dataset):
#      def __init__(self, input_ids, token_type_ids, attention_mask, ids, word2idx, feat2idx,
#                                  src_insts=None, src_feat_insts=None, num_insts=None,
#                                  ans=None, eqns=None):
#         super().__init__(word2idx, feat2idx, ids, src_insts, src_feat_insts, num_insts,ans, eqns)
#         self._input_ids = input_ids
#         self._token_type_ids = token_type_ids
#         self._attention_mask = attention_mask

#      def __len__(self):
#          return self._input_ids.shape[0]
                                                            
#      def __getitem__(self, idx):
#          if self._ans is not None:
#              return self._input_ids[idx], self._token_type_ids[idx], self._attention_mask[idx], self._ids[idx], self._src_insts[idx], self._src_feat_insts[idx],\
#                  self._num_insts[idx], self._ans[idx], self._eqns[idx]
#          return self._input_ids[idx], self._token_type_ids[idx], self._attention_mask[idx], self._ids[idx], self._src_insts[idx], self._src_feat_insts[idx],\
#                  self._num_insts[idx], self._ans[idx], \
#                       None, None

