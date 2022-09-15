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
import copy

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
#writer = SummaryWriter('runs/m23k_eqn_explore')

torch.autograd.set_detect_anomaly(True)


class BeamSearchNode(object):
    def __init__(self, previousNode, pos, logProb, dict_len, dec_len, rewards, pstack, idx):
        self.pos = pos
        self.prevNode = previousNode
        self.logp = logProb
        self.dict_len = dict_len
        self.dec_len = dec_len
        self.rewards = rewards
        self.pstack = pstack
        self.idx = idx

    def eval(self):
        return self.logp / float(self.dec_len + 1e-6)

    def __lt__(self, other):
        if self.idx == -1:
            return False
        elif other.idx == -1:
            return True
        return (self.idx < other.idx)



def train_with_beamsearch(args, dataset, encoder, decoder, device, save_path="models/", beam_width=1):
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
        batch_num=0
        for batch in tqdm(dataset):
            batch_num+=1
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = 0
            ids, src_insts, src_feat_insts, src_lengths, num_insts, num_lengths, num_masks, ans_insts, _ = batch[-9:]
            src_insts = src_insts.to(device)
            src_feat_insts = src_feat_insts.to(device)
            src_lengths = src_lengths.to(device)
            # print("src", src_insts, src_lengths, src_feat_insts)

            if args.encoder == 'gru':
                output, hidden = encoder(src_insts, src_lengths.cpu(), src_feat_insts)

            # if batch_num==3:

                # print("out", output, hidden)


            # print("outp", output, hidden)
            # exit()

            batch_size = num_insts.shape[0]
            operand_dict = num_insts.to(device)
            probability_mask = num_masks.to(device)
            operand_dict = torch.cat((num_insts, torch.zeros(
                (operand_dict.shape[0], args.operand_vocab_size - operand_dict.shape[1])).double()), 1)
            probability_mask = torch.cat((num_masks, torch.zeros(
                (probability_mask.shape[0], args.operand_vocab_size - probability_mask.shape[1])).double()), 1)

            # print(operand_dict[0], operand_dict[1])
            operand_dict = operand_dict.to(device)
            probability_mask = probability_mask.to(device)

            rewards = torch.zeros(0,args.max_decode_length)
            probabilities = torch.zeros(0,3 * args.max_decode_length)
            batch_size = len(num_insts)
            indices = torch.zeros(batch_size).fill_(-1)

            decoder_input = torch.tensor([[START_OP] * batch_size], device=device).T

            num_lengths1 = copy.deepcopy(num_lengths)

            nodes_array = [PriorityQueue() for l in range(batch_size)]

            # print("HIDDEN", hidden.shape)
            for l in range(batch_size):
                dict_len = num_lengths1[l].unsqueeze(0)
                rew = torch.tensor([])
                pstack = torch.tensor([])
                idx = torch.tensor(-1)
                # previousNode, pos, logProb, dict_len, dec_len, rewards, pstack, idx

                node = BeamSearchNode(None, l, 0, dict_len, 0, rew, pstack, idx)
                nodes_array[l].put(node)

            for j in range(args.max_decode_length):
                ans_insts_ = []
                num_lengths_ = torch.tensor([]).long()
                prevNodes = []
                pos_indices = []

                # print("hidden1", hidden.shape)

                for l in range(batch_size):
                    k = 0
                    while not nodes_array[l].empty() and k < beam_width:
                        n = nodes_array[l].get()
                        k += 1
                        # if l<10:
                            # print(l, n.pos)
                        pos_indices.append(n.pos)
                        ans_insts_.append(ans_insts[l])
                        num_lengths_ = torch.cat((num_lengths_,n.dict_len), dim=0)
                        prevNodes.append(n)
                # if batch_num==3:

                    # if j>=1:
                        # print("hidden1", hidden)
                        # print("dec1", decoder_input)

                if j > 0:
                    # print("pos_indices", pos_indices)
                    # print(hidden)
                    # print(decoder_input)
                    hidden = hidden[pos_indices,:]
                    decoder_input = decoder_input[pos_indices,:]
                    operand_dict = operand_dict[pos_indices,:]
                    probability_mask = probability_mask[pos_indices,:]

                # print("hidden2", hidden.shape)
                # if batch_num==3:

                    # if j>=1:
                        # print("hidden", hidden, hidden.shape)
                        # print("decoder_inp", decoder_input, decoder_input.shape)
                # if j==2:
                    # exit()


                new_bsize = decoder_input.size(0)

                operator, op1, op2, hidden, op_id = decoder(decoder_input, hidden, sample=True)

                # print("op_a", op_id.view(-1)[:5], op_id.view(-1).shape)
                # if batch_num==3:

                    # if j>=1:
                        # print("op1", op1)
                        # print("op2", op2)



                op1_prob = MaskedSoftmax()(op1, probability_mask)
                op2_prob = MaskedSoftmax()(op2, probability_mask)

                op_distrib = operator.view(new_bsize,-1,1,1) * op1_prob.view(new_bsize,1,-1,1) * op2_prob.view(new_bsize,1,1,-1)
                
                # if j>=1:
                    # print("opd", op_distrib, op_distrib.shape)
                    # exit()
                sizes = torch.tensor(op_distrib.size()).to(device)
                samples = torch.multinomial(op_distrib.view(new_bsize,-1), beam_width)
                op_id = torch.div(samples, sizes[2] * sizes[3]).long()
                pos = samples - (op_id * sizes[2] * sizes[3])
                op1_id = torch.div(pos, sizes[2]).long()
                op2_id = torch.remainder(pos, sizes[2]).long()

                # if batch_num==3:
                    # if j>=1:
                        # print("op", op_id.view(-1), op_id.view(-1).shape)

                # print("op1", op1_id)

                # print("op2", op2_id)

                # exit()



                probability_stack = torch.stack([operator[torch.tensor(range(new_bsize)).unsqueeze(1), op_id.long()], op1_prob[
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
                probability_stack = probability_stack.cpu()
                nodes_array = [PriorityQueue() for l in range(batch_size)]
                for l in range(new_bsize):
                    for k in range(beam_width):
                        # logp_ = prevNodes[l].logp + logprob[l,k].item()
                        pos_ = l * beam_width + k
                        if op_id[l,k] == 0: rewards_[l,k] = 0
                        rew_ = torch.cat((prevNodes[l].rewards, rewards_[l, k].unsqueeze(0)), dim=0)
                        pstack_ = torch.cat((prevNodes[l].pstack, probability_stack[l, k]), dim=0)
                        idx_ = prevNodes[l].idx
                        if prevNodes[l].idx == -1 and op_id[l,k] == 0: idx_ = torch.tensor(j)
                        if rewards_[l,k] == 1 and idx_ == -1: idx_ = torch.tensor(j)

                        new_node = BeamSearchNode(None, pos_, 0, num_lengths_[l,k].unsqueeze(0), prevNodes[l].dec_len+1,
                                                    rew_, pstack_, idx_)
                        pq_idx = l if j == 0 else (l // beam_width)
                        nodes_array[pq_idx].put(new_node)


            for l in range(batch_size):
                selectedNode = nodes_array[l].get()
                indices[l] = selectedNode.idx
                probabilities = torch.cat([probabilities, selectedNode.pstack.unsqueeze(0)], dim=0)
                rewards = torch.cat([rewards, selectedNode.rewards.unsqueeze(0)], dim=0)
            # if batch_num==3:
                # print("indices", indices)

                # print("probabilities", probabilities)

                # print("rewards", rewards)

                # exit()

            indices.to(device)
            rewards.to(device)
            probabilities.to(device)

            indices[(indices == -1)] = (args.max_decode_length - 1)

            unsupervised_reward = (rewards[range(batch_size), indices.long()]>0).float()
            rewards[range(batch_size), indices.long()] = 2 * rewards[range(batch_size), indices.long()] - 1

            probabilities = probabilities.to(device)
            rewards = rewards.to(device)
            indices = indices.to(device)

            loss, _ = batch_loss_REINFORCE(rewards, probabilities.view(batch_size, -1, 3), indices.int(), device)

            reward = unsupervised_reward.sum().item()

            total_loss += loss
            n_batch_processed += 1
            n_examples_processed += batch_size
            total_reward += reward

            # print(total_reward, n_examples_processed)


            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        encoder_scheduler.step()
        decoder_scheduler.step()

        print("Epoch {}: Loss -  {}  Acc - {}".format(i, total_loss / n_batch_processed,
                                                      total_reward / n_examples_processed))

        #writer.add_scalar('training loss bw8', total_loss / n_batch_processed, i)

        #writer.add_scalar('training accuracy bw8', total_reward / n_examples_processed, i)

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
