import torch
from torch import nn
from constants import *
from tqdm import tqdm
from utils import *
from masked_softmax import MaskedSoftmax
from queue import PriorityQueue

max_queue_size = 200

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, op, op1, op2, op_dict, p_mask, logProb, dict_len, dec_len):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.op_id = op
        self.op1_id = op1
        self.op2_id = op2
        self.logp = logProb
        self.operand_dict = op_dict
        self.probability_mask = p_mask
        self.dict_len = dict_len
        self.dec_len = dec_len

    def eval(self):
        return self.logp / float(self.dec_len + 1e-6)

    def __lt__(self, other):
        return True


def test_with_beamsearch_stop(args, dataset, encoder, decoder, device, beam_width=1):
    encoder.eval()
    decoder.eval()

    examples_processed = 0
    num_correct = 0

    with torch.no_grad():
        for batch in tqdm(dataset):
            ids, src_insts, src_feat_insts, src_lengths, num_insts, num_unit_insts, num_rate_insts, num_lengths, num_masks, ans_insts, ans_unit_insts, ans_rate_insts, _ = batch[-13:]
            src_insts = src_insts.to(device)
            src_feat_insts = src_feat_insts.to(device)
            src_lengths = src_lengths.to(device)
            if args.encoder == 'gru':
                output, hidden = encoder(src_insts, src_lengths, src_feat_insts)
            if args.encoder == 'bert':
                input_ids, token_type_ids, attention_mask = batch[0:3]
                hidden = encoder(input_ids.to(device), token_type_ids.to(device), attention_mask.to(device))

            batch_size = num_insts.shape[0]
            operand_dict = num_insts.to(device)
            probability_mask = num_masks.to(device)
            operand_dict = torch.cat((num_insts, torch.zeros(
                (operand_dict.shape[0], args.operand_vocab_size - operand_dict.shape[1])).double()), 1)
            probability_mask = torch.cat((num_masks, torch.zeros(
                (probability_mask.shape[0], args.operand_vocab_size - probability_mask.shape[1])).double()), 1)

            operand_dict = operand_dict.to(device)
            probability_mask = probability_mask.to(device)

            decoder_input = torch.tensor([[START_OP] * batch_size], device=device).T

            for i in range(batch_size):
                hidden_i = hidden[:,i,:].unsqueeze(1)
                di = decoder_input[i,:].unsqueeze(1)
                operand_dict_i = operand_dict[i,:]
                dict_len = num_lengths[i]
                ans = ans_insts[i]
                probability_mask_i = probability_mask[i,:]

                node = BeamSearchNode(hidden_i, None, di, None, None, operand_dict_i, probability_mask_i, 0, dict_len, 0)
                beam_nodes = PriorityQueue()

                beam_nodes.put((-node.eval(), node))
                qsize = 1
                reward = 0
                num_endnodes = 0

                while qsize > 0 and qsize < max_queue_size:
                    score, n = beam_nodes.get()
                    qsize -= 1
                    if n.dec_len >= args.max_decode_length or n.dict_len >= args.operand_vocab_size: continue

                    if n.op_id.item() == 0 and n.dec_len > 1:
                        num_endnodes += 1
                        prev = n.prevNode
                        old_result = execute_single(prev.op_id.squeeze(0), prev.op1_id, prev.op2_id, prev.operand_dict)
                        if abs((old_result - ans).item()) < 1e-3:
                            reward = 1
                        if num_endnodes >= 1:
                            break
                        else:
                            continue


                    hidden_i = n.h
                    di = n.op_id
                    operand_dict_i = n.operand_dict
                    probability_mask_i = n.probability_mask
                    dict_len = n.dict_len

                    operator, op1, op2, hidden_i, op_id = decoder(di, hidden_i.to(device), sample=False)

                    op1_prob = MaskedSoftmax()(op1, probability_mask_i)
                    op2_prob = MaskedSoftmax()(op2, probability_mask_i)

                    op_distrib = (operator.view(-1,1,1) * op1_prob.view(1,-1,1) * op2_prob.view(1,1,-1))
                    sizes = op_distrib.size()
                    prob, indices = torch.topk(op_distrib.view(1,-1), beam_width)

                    for k in range(beam_width):
                        pos = int(indices[0][k].item())
                        op_id = torch.tensor(pos // int(sizes[1] * sizes[2])).unsqueeze(0).to(device)
                        pos -= op_id.item() * int(sizes[1]) * int(sizes[2])
                        op1_id = torch.tensor(pos // int(sizes[1]))
                        op2_id = torch.tensor(pos % int(sizes[1]))

                        result = execute_single(op_id, op1_id, op2_id, operand_dict_i)

                        operand_dict_i[dict_len] = result.squeeze()
                        vocab_size_add = op_id.clone().cpu()
                        dict_len = n.dict_len + (vocab_size_add.item() > 0)
                        probability_mask_i_new = probability_mask_i.clone()
                        probability_mask_i_new[dict_len - 1] = 1
                        probability_mask_i = probability_mask_i_new.clone()
                        di = op_id.clone().unsqueeze(0)
                        logp = torch.log(prob[0][k]).item()

                        new_node = BeamSearchNode(hidden_i, n, di, op1_id, op2_id, operand_dict_i, probability_mask_i, n.logp+logp, dict_len, n.dec_len+1)
                        score_ = -new_node.eval()

                        beam_nodes.put((score_, new_node))
                        qsize += 1

                if reward >= 1:
                    num_correct += 1

                examples_processed += 1

    return num_correct / examples_processed