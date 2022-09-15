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
'''
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
'''
torch.autograd.set_detect_anomaly(True)

# random.seed(0)

class BeamSearchNode(object):
    def __init__(self, pos, k, idx):
        self.pos = pos
        self.k = k
        self.idx = idx

    # def eval(self):
        # return self.pos

    def __lt__(self, other):
        if self.idx == -1:
            return False
        elif other.idx == -1:
            return True
        return (self.idx < other.idx)



def train(args, dataset, encoder, decoder, device, save_path="models/", beam_width=1):
    n_epochs = args.num_epochs
    encoder_optimizer = Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    encoder_scheduler = lr_scheduler.StepLR(encoder_optimizer, step_size=args.step_size, gamma=args.gamma)

    decoder_optimizer = Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    decoder_scheduler = lr_scheduler.StepLR(decoder_optimizer, step_size=args.step_size, gamma=args.gamma)
    model_save_path = save_path + "epoch.pt"

    # random.seed(0)

    for i in range(n_epochs):
        print("epoch:", i)
        total_reward = 0
        n_examples_processed = 0
        n_batch_processed = 0
        total_loss = 0
        batch_num = 0
        for batch in tqdm(dataset):
            #print(batch_num)
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

            # exit()
            batch_size = num_insts.shape[0]
            #print(batch_size)
            operand_dict = num_insts.to(device)
            probability_mask = num_masks.to(device)
            operand_dict = torch.cat((num_insts, torch.zeros(
                (operand_dict.shape[0], args.operand_vocab_size - operand_dict.shape[1])).double()), 1)
            probability_mask = torch.cat((num_masks, torch.zeros(
                (probability_mask.shape[0], args.operand_vocab_size - probability_mask.shape[1])).double()), 1)

            # print(operand_dict[0], operand_dict[1])
            operand_dict = operand_dict.to(device)
            probability_mask = probability_mask.to(device)

            batch_size = len(num_insts)
            bbsize = batch_size*beam_width
            indices = torch.zeros(batch_size).fill_(-1)

            decoder_input = torch.tensor([[START_OP] * batch_size], device=device).T

            num_lengths1 = copy.deepcopy(num_lengths)

            rewards = torch.tensor([[]] * bbsize).to(device)

            probabilities = torch.tensor([[]] * bbsize).to(device)

            for j in range(args.max_decode_length):


                if j==0:
                    ans_insts_ = torch.tensor(ans_insts)
                    num_lengths_ = num_lengths
                    
                else:
             #       print("j=2", indices_.shape, hidden.shape)

                    if j == 1:
                        bb = beam_width
                    else:
                        bb = beam_width*beam_width
                    indices_ = indices_.view(batch_size, bb)
                    nodes_array = [PriorityQueue() for l in range(batch_size)]

                    for l in range(batch_size):
                        for k in range(bb):
                            new_node = BeamSearchNode(l, k, indices_[l][k].item())
                            nodes_array[l].put(new_node)

                    paths_chosen = torch.zeros((batch_size, beam_width), dtype=torch.int64).to(device)

                    for l in range(batch_size):
                        k = 0
                        while not nodes_array[l].empty() and k < beam_width:
                            n = nodes_array[l].get()
                            paths_chosen[l][k] = n.k
                            k += 1

                    # new
                    rewards = rewards.repeat_interleave(beam_width, dim=0)

                    probabilities = probabilities.repeat_interleave(beam_width, dim=0)

                    rewards = rewards.view(batch_size, beam_width*beam_width, -1)

                    probabilities = probabilities.view(batch_size, beam_width*beam_width, -1)

                    rewards = torch.gather(rewards, 1, paths_chosen.unsqueeze(2).repeat(1,1,rewards.shape[2])).view(batch_size*beam_width, -1)

                    probabilities = torch.gather(probabilities, 1, paths_chosen.unsqueeze(2).repeat(1,1,probabilities.shape[2])).view(batch_size*beam_width, -1)

                    rewards_ = torch.gather(rewards_.view(batch_size, bb), 1, paths_chosen)
                    indices = torch.gather(indices_, 1, paths_chosen).view(-1)
                    probability_stack = torch.gather(probability_stack.view(batch_size, bb, 3), 1, paths_chosen.unsqueeze(2).repeat(1,1,3))
                    rewards = torch.cat([rewards, rewards_.view(-1).unsqueeze(1)], dim=-1)
                    probabilities = torch.cat([probabilities, probability_stack.view(-1,3)], dim=-1)

                    hidden = hidden.view(batch_size, bb, -1)
                    hidden = torch.gather(hidden, 1, paths_chosen.unsqueeze(2).repeat(1,1,hidden.shape[2])).view(batch_size*beam_width, -1)
                    result = torch.gather(result.view(batch_size, bb), 1, paths_chosen).view(-1)

                    op_id = torch.gather(op_id.view(batch_size, bb), 1, paths_chosen).view(-1)
                    decoder_input = op_id.clone().detach().T
                    # print("j=2", paths_chosen.shape, operand_dict.shape, probability_mask.shape)

                    operand_dict = torch.gather(operand_dict.view(-1, bb, args.operand_vocab_size), 1, paths_chosen.unsqueeze(2).repeat(1,1,args.operand_vocab_size)).view(-1, args.operand_vocab_size)


                    probability_mask = torch.gather(probability_mask.view(batch_size, bb, -1), 1, paths_chosen.unsqueeze(2).repeat(1,1,args.operand_vocab_size)).view(batch_size*beam_width, -1)

                    ans_insts_ = torch.tensor(ans_insts).repeat_interleave(beam_width, dim=0)
                    
                    num_lengths_ = torch.gather(num_lengths_.view(batch_size, bb).to(device), 1, paths_chosen).view(-1).cpu()


                new_bsize = decoder_input.size(0)

                # print("dec, hid", decoder_input.shape, hidden.shape)
                decoder_input = decoder_input.view(new_bsize, -1)

                operator, op1, op2, hidden, op_id = decoder(decoder_input, hidden, sample=True)

                op1_prob = MaskedSoftmax()(op1, probability_mask)
                op2_prob = MaskedSoftmax()(op2, probability_mask)

                op_distrib = operator.view(new_bsize,-1,1,1) * op1_prob.view(new_bsize,1,-1,1) * op2_prob.view(new_bsize,1,1,-1)

                sizes = torch.tensor(op_distrib.size()).to(device)
                samples = torch.multinomial(op_distrib.view(new_bsize,-1), beam_width)
                op_id = torch.div(samples, sizes[2] * sizes[3]).long()
                pos = samples - (op_id * sizes[2] * sizes[3])
                op1_id = torch.div(pos, sizes[2]).long()
                op2_id = torch.remainder(pos, sizes[2]).long()


                probability_stack = torch.stack([operator[torch.tensor(range(new_bsize)).unsqueeze(1), op_id.long()], op1_prob[
                    torch.tensor(range(new_bsize)).unsqueeze(1), op1_id], op2_prob[torch.tensor(range(new_bsize)).unsqueeze(1), op2_id]], dim=-1)
                result = execute_supervised_beam(op_id, op1_id, op2_id, operand_dict)
              #  print(result.shape)
              #  print(ans_insts_.shape)
                rewards_ = (torch.abs(result - ans_insts_.unsqueeze(1).double().to(device)) < 1e-3).float()
                #print(rewards_.shape)
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


                indices_ = indices.unsqueeze(1).repeat(1, beam_width).to(device)

                indices_[(op_id == 0) & (indices_ == -1)] = j
               # print(indices_.shape)
                #print(rewards_.shape)
                #exit()
                indices_[(rewards_==1) & (indices_ == -1)] = j

                if j == 0: rewards_[(op_id == 0)] = 0

            indices_ = indices_.view(batch_size, beam_width*beam_width)

            nodes_array = [PriorityQueue() for l in range(batch_size)]

            for l in range(batch_size):

                for k in range(beam_width*beam_width):

                    new_node = BeamSearchNode(l, k, indices_[l][k].item())

                    nodes_array[l].put(new_node)

            paths_chosen = torch.zeros((batch_size, 1), dtype=torch.int64).to(device)


            for l in range(batch_size):

                selectedNode = nodes_array[l].get()

                paths_chosen[l,0] = selectedNode.k

            rewards_ = torch.gather(rewards_.view(batch_size, beam_width*beam_width), 1, paths_chosen)

            indices = torch.gather(indices_, 1, paths_chosen).view(-1)

            probability_stack = torch.gather(probability_stack.view(batch_size, beam_width*beam_width, 3), 1, paths_chosen.unsqueeze(2).repeat(1,1,3))

            rewards = rewards.repeat_interleave(beam_width, 0).view(batch_size, beam_width*beam_width, -1)

            probabilities = probabilities.repeat_interleave(beam_width, 0).view(batch_size, beam_width*beam_width, -1)

            rewards = torch.gather(rewards, 1, paths_chosen.unsqueeze(2).repeat(1,1,args.max_decode_length-1)).squeeze(1)

            probabilities = torch.gather(probabilities, 1, paths_chosen.unsqueeze(2).repeat(1,1,(args.max_decode_length-1)*3)).squeeze(1)

            rewards = torch.cat([rewards, rewards_.view(-1).unsqueeze(1)], dim=-1)

            probabilities = torch.cat([probabilities, probability_stack.view(-1,3)], dim=-1)

            indices[(indices == -1)] = (args.max_decode_length - 1)

            unsupervised_reward = (rewards[range(batch_size), indices.long()]>0).float()
            #print("rewards:", rewards[0])
            rewards[range(batch_size), indices.long()] = 2 * rewards[range(batch_size), indices.long()] - 1
            #print("new rewards:", rewards[0])
            unsupervised_probabilities = probabilities

            unsupervised_rewards = rewards

            unsupervised_indices = indices

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
