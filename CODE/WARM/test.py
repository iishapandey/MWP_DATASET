import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from constants import *
from tqdm import tqdm
from utils import *  
from masked_softmax import MaskedSoftmax
from data_preprocess import Et
import numpy as np
import copy
import json

# torch.autograd.set_detect_anomaly(True)

def loss_REINFORCE(rewards, probabilities, device):
	loss = 0
	n = len(rewards)
	cumulative_reward = [0]
	for i in range(n):
		cumulative_reward.append(rewards[n - i - 1].squeeze() + cumulative_reward[-1])
		# print("n", i, probabilities[n - i - 1])
		loss = loss + torch.log(probabilities[n - i - 1]) * cumulative_reward[-1]
	return loss.mean() / n


def batch_loss_REINFORCE(rewards, probabilities, indices, device):
	loss, reward = 0, 0
	bsz = len(rewards)
	for i in range(bsz):
		# print("part", loss)
		# print(i)
		loss += loss_REINFORCE(rewards[i][:indices[i] + 1], probabilities[i][:indices[i] + 1], device).float()
		reward += rewards[i][indices[i]].float()
	loss = - loss / bsz
	return loss, reward

def test_with_beamsearch(args, dataset, encoder, decoder, device):
	beam_width = 20
	encoder.eval()
	decoder.eval()

	print("In test.py")
	# 2) Vectorised Code

	total_reward = 0
	n_examples_processed = 0
	n_batch_processed = 0
	total_loss = 0
	batch_num=0

	all_eqs = []

	with open('data/math23k/questions/train.json', 'r', encoding='utf-8') as f:
		train_data = json.load(f)

	var_names = [[qu[0] for qu in d['quantity-units']] for d in train_data]
	values = [[qu[1] for qu in d['quantity-units']] for d in train_data]

	for i in range(len(values)):
		vals = values[i]
		vnames = var_names[i]
		if 1.0 not in vals:
			vnames.append('1.0')
		if 3.14 not in vals:
			vnames.append('3.14')



	with torch.no_grad():
		for batch in tqdm(dataset):
			batch_num+=1
			loss = 0
			ids, src_insts, src_feat_insts, src_lengths, num_insts, num_unit_insts, num_rate_insts, num_lengths, num_masks, ans_insts, ans_unit_insts, ans_rate_insts, eqn_insts = batch[-13:]

			# annot_insts = torch.tensor(annot_insts).to(device)
			
			# unsupervised part

			src_insts = src_insts.to(device)
			src_feat_insts = src_feat_insts.to(device)

			num_lengths1 = copy.deepcopy(num_lengths).repeat_interleave(beam_width, dim=0).to(device)

			if args.encoder == 'gru':
				output, hidden = encoder(src_insts, src_lengths, src_feat_insts)
			if args.encoder == 'bert':
				input_ids, token_type_ids, attention_mask = batch[0:3]
				hidden = encoder(input_ids.to(device), token_type_ids.to(device), attention_mask.to(device))

			batch_size = num_insts.shape[0]
			bbsize = batch_size * beam_width
			operand_dict = num_insts.to(device)
			probability_mask = num_masks.to(device)
			operand_dict = torch.cat((num_insts, torch.zeros(
				(operand_dict.shape[0], args.operand_vocab_size - operand_dict.shape[1])).double()), 1)
			probability_mask = torch.cat((num_masks, torch.zeros(
				(probability_mask.shape[0], args.operand_vocab_size - probability_mask.shape[1])).double()), 1)

			var_names_batch = var_names[:batch_size]
			var_names = var_names[batch_size:]

			var_names_batch = [var+['']*(args.operand_vocab_size-len(var)) for var in var_names_batch]

			var_names_batch1 = [[var]*beam_width for var in var_names_batch]

			var_names_batch1 = [v for varlist in var_names_batch1 for v in varlist]

			eqntree_dict = [[Et(num) for num in operand] for operand in var_names_batch1]

			# eqntree_dict = [[Et(num.item()) for num in operand] for operand in operand_dict.repeat_interleave(beam_width, dim=0)]

			operand_dict = operand_dict.repeat_interleave(beam_width, dim=0)
			probability_mask = probability_mask.repeat_interleave(beam_width, dim=0)

			operand_dict = operand_dict.to(device)
			probability_mask = probability_mask.to(device)

			rewards = torch.tensor([[]] * bbsize).to(device)
			probabilities = torch.tensor([[]] * bbsize).to(device)

			indices = torch.zeros(bbsize).fill_(-1).to(device)

			decoder_input = torch.tensor([[START_OP] * bbsize], device=device).T
			# print(hidden.shape)
			hidden = hidden.repeat_interleave(beam_width, dim=1)

			# print(ans_insts)
			ans_insts = torch.tensor(list(ans_insts)).repeat_interleave(beam_width, dim=0).to(device)
			num_lengths2 = copy.deepcopy(num_lengths)

			result = torch.tensor([0])			

			for j in range(args.max_decode_length):

				new_bsize = decoder_input.size(0)
				# print(new_bsize)

				operator, op1, op2, hidden, op_id = decoder(decoder_input, hidden, sample=True)

				op1_prob = MaskedSoftmax()(op1, probability_mask)
				op2_prob = MaskedSoftmax()(op2, probability_mask)

				op_distrib = operator.view(new_bsize,-1,1,1) * op1_prob.view(new_bsize,1,-1,1) * op2_prob.view(new_bsize,1,1,-1)

				sizes = torch.tensor(op_distrib.size()).to(device)
				samples = torch.multinomial(op_distrib.view(new_bsize,-1), beam_width, replacement=True)
				op_id = torch.div(samples, sizes[2] * sizes[3]).long()
				pos = samples - (op_id * sizes[2] * sizes[3])
				op1_id = torch.div(pos, sizes[2]).long()
				op2_id = torch.remainder(pos, sizes[2]).long()


				###################################################

				eqntree_dict_beam = []

				# print(batch_size*beam_width)
				# print("Decoding step ",j)

				operand_dict_beam = []

				for b1 in range(batch_size*beam_width):
					for b2 in range(beam_width):
						eqtree_dict_b=copy.deepcopy(eqntree_dict[b1])
						t = Et(OPERATOR_DICT[op_id[b1][b2].item()+1])
						t.left = eqtree_dict_b[op1_id[b1][b2].item()]
						t.right = eqtree_dict_b[op2_id[b1][b2].item()]
						eqtree_dict_b[num_lengths1[b1]] = t
						eqntree_dict_beam.append(eqtree_dict_b)
						operand_dict_beam.append(operand_dict[b1])

				eqntree_dict_beam = [eqntree_dict_beam[i:i + beam_width*beam_width] for i in range(0, len(eqntree_dict_beam), beam_width*beam_width)]

				operand_dict_beam1 = copy.deepcopy(operand_dict_beam)

				operand_dict_beam = [operand_dict_beam[i:i + beam_width*beam_width] for i in range(0, len(operand_dict_beam), beam_width*beam_width)]

				########################################################
				
				num_lengths_ = copy.deepcopy(num_lengths1).repeat_interleave(beam_width, dim=0).to(device)

				probability_stack = torch.stack([operator[torch.tensor(range(new_bsize)).unsqueeze(1), op_id], op1_prob[
			    	torch.tensor(range(new_bsize)).unsqueeze(1), op1_id], op2_prob[torch.tensor(range(new_bsize)).unsqueeze(1), op2_id]], dim=-1)

				result = execute_supervised_beam(op_id, op1_id, op2_id, operand_dict)

				result = result.view(batch_size*beam_width, -1)
				ans_insts_ = ans_insts.unsqueeze(1).repeat(1,beam_width)

				rewards_ = (torch.abs(result - ans_insts_.double().to(device)) < 1e-3).float()


				indices_ = indices.unsqueeze(1).repeat(1, beam_width)
				# print(indices_.shape)

				indices_[(op_id == 0) & (indices_ == -1)] = j

				if j == 0: rewards_[(op_id == 0)] = 0

				indices_[(rewards_==1) & (indices_ == -1)] = j

				# print(indices_.shape)

				indices_ = indices_.view(batch_size, beam_width*beam_width)

				priorities = (indices_!=-1)*(indices_.amax(axis=1).unsqueeze(1)-indices_) + (indices_==-1)*indices_ # choose top k priorities (the eqns that terminated the fastest are those with higher priority)
				
				paths_chosen = torch.topk(priorities, beam_width, dim=1).indices.to(device)
				


				########################################
				eqntree_dict = []
				operand_dict = []

				for b1 in range(batch_size):
					for b2 in range(beam_width):
						eqntree_dict.append(eqntree_dict_beam[b1][paths_chosen[b1][b2].item()])
						operand_dict.append(operand_dict_beam[b1][paths_chosen[b1][b2].item()])
				##########################################



				rewards_ = torch.gather(rewards_.view(batch_size, beam_width*beam_width), 1, paths_chosen)
				indices = torch.gather(indices_, 1, paths_chosen).view(-1)
				# if j<5:	
				# 	print("indices 2", indices_[47,:])
				num_lengths1 = torch.gather(num_lengths_.view(batch_size, beam_width*beam_width), 1, paths_chosen).view(-1)

				# print(indices.view(batch_size, -1) == indices1)
				probability_stack = torch.gather(probability_stack.view(batch_size, beam_width*beam_width, 3), 1, paths_chosen.unsqueeze(2).repeat(1,1,3))

				rewards = torch.cat([rewards, rewards_.view(-1).unsqueeze(1)], dim=-1)
				probabilities = torch.cat([probabilities, probability_stack.view(-1,3)], dim=-1)

				hidden = hidden.unsqueeze(1).repeat(1, beam_width, 1).view(batch_size, beam_width*beam_width, -1)
				hidden = torch.gather(hidden, 1, paths_chosen.unsqueeze(2).repeat(1,1,hidden.shape[2])).view(batch_size*beam_width, -1)


				result = torch.gather(result.view(batch_size, beam_width*beam_width), 1, paths_chosen).view(-1)
				op_id = torch.gather(op_id.view(batch_size, beam_width*beam_width), 1, paths_chosen).view(-1)

				#################################
				operand_dict = torch.stack(operand_dict).to(device)
				################################

				operand_dict[range(len(operand_dict)), num_lengths1] = result.squeeze().detach()
				###################################
				# if j<5:

				vocab_size_add = op_id.clone().detach().cpu()
				vocab_size_add = torch.where(vocab_size_add > 0, torch.tensor(1), vocab_size_add).squeeze().long()
				num_lengths1 += vocab_size_add.to(device)
				probability_mask_new = probability_mask.clone()
				probability_mask_new[range(len(num_lengths1)), num_lengths1 - 1] = 1
				probability_mask = probability_mask_new.clone()
				decoder_input = op_id.clone().detach().T

			indices = indices.view(batch_size, beam_width)
			rewards = rewards.view(batch_size, beam_width, -1)
			probabilities = probabilities.view(batch_size, beam_width, -1)
			
			#########################
			eqntree_dict = [eqntree_dict[i:i + beam_width] for i in range(0, len(eqntree_dict), beam_width)]
			#########################

			priorities = (indices!=-1)*(indices.amax(axis=1).unsqueeze(1)-indices) + (indices==-1)*indices

			ind = torch.topk(priorities, 1, dim=1).indices
			# print(ind.shape)

			indices = torch.gather(indices, 1, ind).squeeze(1)
			rewards = torch.gather(rewards, 1, ind.unsqueeze(2).repeat(1,1,15)).squeeze(1)
			probabilities = torch.gather(probabilities, 1, ind.unsqueeze(2).repeat(1,1,45)).squeeze(1)

			##############################
			operand_dict = [operand_dict[i:i + beam_width] for i in range(0, len(operand_dict), beam_width)]

			eqntree_dict1 = []
			opd_dict = []
			for b1 in range(batch_size):
				opd_dict.append(operand_dict[b1][ind[b1][0].item()])
				eqntree_dict1.append(eqntree_dict[b1][ind[b1][0].item()])
			eqntree_dict = eqntree_dict1
			#################################
					
			num_lengths1 = num_lengths1.view(batch_size, -1)
			num_lengths1 = torch.gather(num_lengths1, 1, torch.topk(priorities, 1, dim=1).indices).squeeze(1)

			ans_insts = ans_insts.view(batch_size, -1)
			ans_insts = torch.gather(ans_insts, 1, torch.topk(priorities, 1, dim=1).indices).squeeze(1)

			indices[(indices == -1)] = (args.max_decode_length - 1)

			unsupervised_reward = (rewards[range(batch_size), indices.long()]>0).float()
			rewards[range(batch_size), indices.long()] = 2 * rewards[range(batch_size), indices.long()] - 1

			##################
			final_predicted_trees = [eqntree_dict[b][indices.long()[b] + num_lengths[b]] for b in range(batch_size)]
			##################


			#########################

			for b in range(batch_size):
				all_eqs.append({"id": ids[b], "pred_eq": final_predicted_trees[b].print_eqn(final_predicted_trees[b]), "true_eq": eqn_insts[b], "correctness": unsupervised_reward[b].item()})

			############################

			final_reward = unsupervised_reward.sum().item() # to calculate accuracy

			loss, _ = batch_loss_REINFORCE(rewards, probabilities.view(batch_size, -1, 3), indices.int(), device)

			# compute net loss 

			total_loss += loss
			n_batch_processed += 1
			n_examples_processed += batch_size
			total_reward += final_reward

			# exit()

		with open("generated_eqns_bs_tensorized_bw20_acc92_expr.json", 'w') as f:
			json.dump(all_eqs, f, ensure_ascii=False, indent=2)

		if args.mode == 'test':
			print(total_reward / n_examples_processed) 

		print("Test Loss -  {} Acc - {}".format(total_loss / n_batch_processed, total_reward / n_examples_processed))


