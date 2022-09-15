import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from constants import *
from tqdm import tqdm
from utils import *
from masked_softmax import MaskedSoftmax
from train import loss_REINFORCE, batch_loss_REINFORCE
from queue import PriorityQueue
import random
'''
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
'''
torch.autograd.set_detect_anomaly(True)

class BeamSearchNode(object):
    def __init__(self, previousNode, pos, logProb, dict_len, dec_len, rewards, pstack, idx, stop_pstack):
        self.pos = pos
        self.prevNode = previousNode
        self.logp = logProb
        self.dict_len = dict_len
        self.dec_len = dec_len
        self.rewards = rewards
        self.pstack = pstack
        self.idx = idx
        self.stop_pstack = stop_pstack

    def eval(self):
        return self.logp / float(self.dec_len + 1e-6)

    def __lt__(self, other):
        if self.idx == -1:
            return False
        elif other.idx == -1:
            return True
        return (self.idx < other.idx)



def train_with_beamsearch_stop(args, dataset, encoder, decoder, device, save_path="models/", beam_width=1):
    n_epochs = args.num_epochs
    encoder_optimizer = Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    encoder_scheduler = lr_scheduler.StepLR(encoder_optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.encoder == 'bert':
        encoder_optimizer = AdamW(encoder.parameters(), lr=args.lr, eps=args.adam_epsilon)
        encoder_scheduler = get_linear_schedule_with_warmup(encoder_optimizer, num_warmup_steps=args.warmup,
                                                            num_training_steps=(1500 / args.batch_size) * args.num_epochs)
    decoder_optimizer = Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    decoder_scheduler = lr_scheduler.StepLR(decoder_optimizer, step_size=args.step_size, gamma=args.gamma)
    model_save_path = save_path + "epoch.pt"

    for i in range(n_epochs):
        total_reward = 0
        n_examples_processed = 0
        n_batch_processed = 0
        total_loss = 0
        for batch in tqdm(dataset):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = 0
           # ids, src_insts, src_feat_insts, src_lengths, num_insts, num_unit_insts, num_rate_insts, num_lengths, num_masks, ans_insts, ans_unit_insts, ans_rate_insts, _ = batch[-13:]
            ids, src_insts, src_feat_insts, src_lengths, num_insts, num_lengths, num_masks, ans_insts, _ = batch[-8:]

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

            rewards = torch.zeros(0,args.max_decode_length)
            probabilities = torch.zeros(0,3 * args.max_decode_length)
            stop_probabilities = torch.zeros(0,args.max_decode_length)
            batch_size = len(num_insts)
            indices = torch.zeros(batch_size).fill_(-1)

            decoder_input = torch.tensor([[START_OP] * batch_size], device=device).T

            nodes_array = [PriorityQueue() for l in range(batch_size)]
            for l in range(batch_size):
                dict_len = num_lengths[l].unsqueeze(0)
                rew = torch.tensor([])
                stop_pstack = torch.tensor([])
                pstack = torch.tensor([])
                idx = torch.tensor(-1)

                node = BeamSearchNode(None, l, 0, dict_len, 0, rew, pstack, idx, stop_pstack)
                nodes_array[l].put(node)

            for j in range(args.max_decode_length):
                ans_insts_ = []
                num_lengths_ = torch.tensor([]).long()
                prevNodes = []
                pos_indices = []

                for l in range(batch_size):
                    k = 0
                    while not nodes_array[l].empty() and k < beam_width:
                        n = nodes_array[l].get()
                        k += 1
                        pos_indices.append(n.pos)
                        ans_insts_.append(ans_insts[l])
                        num_lengths_ = torch.cat((num_lengths_,n.dict_len), dim=0)
                        prevNodes.append(n)

                if j > 0:
                    hidden = hidden[pos_indices,:]
                    decoder_input = decoder_input[pos_indices,:]
                    operand_dict = operand_dict[pos_indices,:]
                    probability_mask = probability_mask[pos_indices,:]

                new_bsize = decoder_input.size(0)

                operator, op1, op2, hidden, op_id = decoder(decoder_input, hidden, sample=True)

                op1_prob = MaskedSoftmax()(op1, probability_mask)
                op2_prob = MaskedSoftmax()(op2, probability_mask)

                op_distrib = operator.view(new_bsize,-1,1,1) * op1_prob.view(new_bsize,1,-1,1) * op2_prob.view(new_bsize,1,1,-1)
                sizes = torch.tensor(op_distrib.size()).to(device)
                samples = torch.multinomial(op_distrib.view(new_bsize,-1), beam_width)
                op_id = torch.div(samples, sizes[2] * sizes[3])
                pos = samples - (op_id * sizes[2] * sizes[3])
                op1_id = torch.div(pos, sizes[2])
                op2_id = torch.remainder(pos, sizes[2]).long()


                probability_stack = torch.stack([operator[torch.tensor(range(new_bsize)).unsqueeze(1), op_id], op1_prob[
                    torch.tensor(range(new_bsize)).unsqueeze(1), op1_id], op2_prob[torch.tensor(range(new_bsize)).unsqueeze(1), op2_id]], dim=-1)
                result = execute_supervised_beam(op_id, op1_id, op2_id, operand_dict)
                # logprob = torch.log(probability_stack[:,:,0] * probability_stack[:,:,1] * probability_stack[:, :, 2]).cpu()

                rewards_ = (torch.abs(result - torch.tensor(ans_insts_).unsqueeze(1).double().to(device)) < 1e-3).float()

                operand_dict = operand_dict.unsqueeze(1).repeat(1,beam_width,1)
                operand_dict[torch.tensor(range(new_bsize)).unsqueeze(1), torch.tensor(range(beam_width)).unsqueeze(0), num_lengths_.unsqueeze(1)] = result.detach()
                vocab_size_add = op_id.clone().detach().cpu()
                vocab_size_add = torch.where(vocab_size_add > 0, torch.tensor(1), vocab_size_add).long()
                num_lengths_ = num_lengths_.unsqueeze(1) + vocab_size_add
                probability_mask_new = probability_mask.unsqueeze(1).repeat(1,beam_width,1).clone()
                probability_mask_new[torch.tensor(range(new_bsize)).unsqueeze(1), torch.tensor(range(beam_width)).unsqueeze(0), num_lengths_ - 1] = 1
                probability_mask = probability_mask_new.clone()

                decoder_input = op_id.clone().detach().view(-1).unsqueeze(1)
                hidden = hidden.repeat_interleave(beam_width, dim=0)
                operand_dict = operand_dict.view(-1, args.operand_vocab_size)
                probability_mask = probability_mask.view(-1, args.operand_vocab_size)

                rewards_ = rewards_.cpu()
                operator = operator.cpu()
                probability_stack = probability_stack.cpu()
                nodes_array = [PriorityQueue() for l in range(batch_size)]
                for l in range(new_bsize):
                    for k in range(beam_width):
                        # logp_ = prevNodes[l].logp + logprob[l,k].item()
                        pos_ = l * beam_width + k
                        if op_id[l,k] == 0: rewards_[l,k] = 0
                        rew_ = torch.cat((prevNodes[l].rewards, rewards_[l,k].unsqueeze(0)), dim=0)
                        pstack_ = torch.cat((prevNodes[l].pstack, probability_stack[l,k]), dim=0)
                        stop_pstack_ = torch.cat((prevNodes[l].stop_pstack, operator[l,0].unsqueeze(0)), dim=0)
                        idx_ = prevNodes[l].idx
                        if prevNodes[l].idx == -1 and op_id[l,k] == 0: idx_ = torch.tensor(j)
                        if rewards_[l,k] == 1 and idx_ == -1: idx_ = torch.tensor(j)

                        new_node = BeamSearchNode(None, pos_, 0, num_lengths_[l,k].unsqueeze(0), prevNodes[l].dec_len+1,
                                                  rew_, pstack_, idx_, stop_pstack_)
                        pq_idx = l if j == 0 else (l // beam_width)
                        nodes_array[pq_idx].put(new_node)

            for l in range(batch_size):
                selectedNode = nodes_array[l].get()
                indices[l] = selectedNode.idx
                probabilities = torch.cat([probabilities, selectedNode.pstack.unsqueeze(0)], dim=0)
                stop_probabilities = torch.cat([stop_probabilities, selectedNode.stop_pstack.unsqueeze(0)], dim=0)
                rewards = torch.cat([rewards, selectedNode.rewards.unsqueeze(0)], dim=0)

            indices.to(device)
            rewards.to(device)
            probabilities.to(device)

            indices[(indices == -1)] = (args.max_decode_length - 1)
            stop_indices = indices[(indices != args.max_decode_length - 1)] + 1
            stop_prob = stop_probabilities[(indices != args.max_decode_length - 1)][range(len(stop_indices)), stop_indices.long()]
            stop_loss = - torch.log(stop_prob).mean()

            reward = rewards[range(batch_size), indices.long()].sum().item()
            rewards[range(batch_size), indices.long()] = 2 * rewards[range(batch_size), indices.long()] - 1

            loss, _ = batch_loss_REINFORCE(rewards, probabilities.view(batch_size, -1, 3), indices.int(), device)
            loss += stop_loss

            total_loss += loss
            n_batch_processed += 1
            n_examples_processed += batch_size
            total_reward += reward

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        encoder_scheduler.step()
        decoder_scheduler.step()

        print("Epoch {}: Loss -  {}  Acc - {}".format(i, total_loss / n_batch_processed,
                                                      total_reward / n_examples_processed))


        checkpoint = {
            'args':args,
            'epoch': i,
            'loss': total_loss / n_batch_processed,
            'acc': total_reward / n_examples_processed,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
        }

        if (i % args.save_freq) == 0:
            model_save_path = save_path + "epoch - " + str(i) + ".pt"
            torch.save(checkpoint, model_save_path)
