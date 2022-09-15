import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from constants import *
from tqdm import tqdm
from utils import *
from masked_softmax import MaskedSoftmax
import numpy as np

# from transformers import (
#     WEIGHTS_NAME,
#     AdamW,
#     get_linear_schedule_with_warmup,
#     squad_convert_examples_to_features,
# )

torch.autograd.set_detect_anomaly(True)


def loss_REINFORCE(rewards, probabilities, device):
    loss = 0
    n = len(rewards)
    cumulative_reward = [0]
    for i in range(n):
        cumulative_reward.append(rewards[n - i - 1].squeeze() + cumulative_reward[-1])
        loss = loss + torch.log(probabilities[n - i - 1]) * cumulative_reward[-1]
    return loss.mean() / n


def batch_loss_REINFORCE(rewards, probabilities, indices, device):
    loss, reward = 0, 0
    bsz = len(rewards)
    for i in range(bsz):
        loss += loss_REINFORCE(rewards[i][:indices[i] + 1], probabilities[i][:indices[i] + 1], device).float()
        reward += rewards[i][indices[i]].float()
    loss = - loss / bsz
    return loss, reward


def train_kb(args, dataset, encoder, decoder, device, save_path="models/"):
    n_epochs = args.num_epochs
    encoder_optimizer = Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    encoder_scheduler = lr_scheduler.StepLR(encoder_optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.encoder == 'bert':
        encoder_optimizer = AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_epsilon)
        encoder_scheduler = get_linear_schedule_with_warmup(encoder_optimizer, num_warmup_steps=args.warmup, num_training_steps=len(dataset)*args.num_epochs)
    decoder_optimizer = Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    decoder_scheduler = lr_scheduler.StepLR(decoder_optimizer, step_size=args.step_size, gamma=args.gamma)
    model_save_path = save_path + "epoch.pt"

    if args.mode == 'test':
        encoder.eval()
        decoder.eval()

    for i in range(n_epochs):
        total_reward = 0
        n_examples_processed = 0
        n_batch_processed = 0
        total_loss = 0
        for batch in tqdm(dataset):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = 0
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

            unit_operand_dict = num_unit_insts
            unit_op_filler = np.zeros((unit_operand_dict.shape[0], args.operand_vocab_size - unit_operand_dict.shape[1]), dtype=Unit)
            unit_op_filler[:] = Unit('NO')
            unit_rate_dict = num_rate_insts
            unit_rate_filler = np.zeros((unit_rate_dict.shape[0], args.operand_vocab_size - unit_rate_dict.shape[1]), dtype=bool)

            probability_mask = num_masks.to(device)
            operand_dict = torch.cat((num_insts, torch.zeros(
                (operand_dict.shape[0], args.operand_vocab_size - operand_dict.shape[1])).double()), 1)
            probability_mask = torch.cat((num_masks, torch.zeros(
                (probability_mask.shape[0], args.operand_vocab_size - probability_mask.shape[1])).double()), 1)

            unit_operand_dict = np.concatenate((num_unit_insts, unit_op_filler), 1)
            unit_rate_dict = np.concatenate((num_rate_insts, unit_rate_filler), 1)

            operand_dict = operand_dict.to(device)
            probability_mask = probability_mask.to(device)

            rewards = torch.tensor([[]] * batch_size).to(device)
            partial_rewards = torch.tensor([[]] * batch_size).to(device)
            probabilities = torch.tensor([[]] * batch_size).to(device)
            batch_size = len(num_insts)
            indices = torch.zeros(batch_size).fill_(-1).to(device)

            decoder_input = torch.tensor([[START_OP] * batch_size], device=device).T
            result = torch.tensor([0])

            for j in range(args.max_decode_length):
                if args.mode == 'train':
                    operator, op1, op2, hidden, op_id = decoder(decoder_input, hidden.to(device), sample=True)
                elif args.mode == 'test':
                    operator, op1, op2, hidden, op_id = decoder(decoder_input, hidden.to(device), sample=False)
                
                op1_prob = MaskedSoftmax()(op1, probability_mask)
                op2_prob = MaskedSoftmax()(op2, probability_mask)

                if args.mode == 'train':
                    op1_id = torch.multinomial(op1_prob, 1).T.squeeze()
                    op2_id = torch.multinomial(op2_prob, 1).T.squeeze()

                elif args.mode == 'test':
                    op1_id = torch.argmax(op1_prob, 1)
                    op2_id = torch.argmax(op2_prob, 1)

                probability_stack = torch.stack([operator[range(batch_size), op_id.squeeze()], op1_prob[
                        range(batch_size), op1_id], op2_prob[range(batch_size), op2_id]], dim=-1)
                probabilities = torch.cat([probabilities, probability_stack], dim=-1)
                result = execute_supervised(op_id, op1_id, op2_id, operand_dict)
                partial_reward, unit_result = execute_unit_supervised(op_id, op1_id, op2_id, unit_operand_dict)
                    
                rewards = torch.cat([rewards, (
                            torch.abs(result - torch.tensor(ans_insts).double().to(device)) < 1e-3).float().unsqueeze(
                    -1)], dim=-1)
                partial_rewards = torch.cat([partial_rewards, torch.tensor(partial_reward).float().to(device).unsqueeze(-1)], dim=-1)

                indices[(op_id.squeeze() == 0) & (indices == -1)] = j

                if j == 0: rewards[(op_id.squeeze() == 0), j] = 0
                indices[(rewards[range(batch_size), -1] == 1) & (indices == -1)] = j

                operand_dict[range(len(operand_dict)), num_lengths] = result.squeeze().detach()
                unit_operand_dict[range(len(operand_dict)), num_lengths] = unit_result.squeeze()
                vocab_size_add = op_id.clone().detach().cpu()
                vocab_size_add = torch.where(vocab_size_add > 0, torch.tensor(1), vocab_size_add).squeeze().long()
                num_lengths += vocab_size_add
                probability_mask_new = probability_mask.clone()
                probability_mask_new[range(len(num_lengths)), num_lengths - 1] = 1
                probability_mask = probability_mask_new.clone()
                decoder_input = op_id.clone().detach().T

            indices[(indices == -1)] = (args.max_decode_length - 1)

            reward = rewards[range(batch_size), indices.long()].sum().item()
            rewards[range(batch_size), indices.long()] = 100*(2 * rewards[range(batch_size), indices.long()] - 1)
            rewards += partial_rewards

            loss, _ = batch_loss_REINFORCE(rewards, probabilities.view(batch_size, -1, 3), indices.int(), device)

            total_loss += loss
            n_batch_processed += 1
            n_examples_processed += batch_size
            total_reward += reward
            if args.mode == 'train':
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

        if args.mode == 'train':
            encoder_scheduler.step()
            decoder_scheduler.step()

        if args.mode == 'test':
            return (total_reward / n_examples_processed)

        print("Epoch {}: Loss -  {}  Acc - {}".format(i, total_loss / n_batch_processed,
                                                      total_reward / n_examples_processed))

        checkpoint = {
            'epoch': i,
            'loss': total_loss / n_batch_processed,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
        }

        if (i%args.save_freq) == 0:
            model_save_path = save_path + "epoch - " + str(i) + ".pt"
            torch.save(checkpoint, model_save_path)
