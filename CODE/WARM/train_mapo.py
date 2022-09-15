from numpy import unicode_
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from constants import *
from tqdm import tqdm
from utils import *
from masked_softmax import MaskedSoftmax
from data_preprocess import Et

import matplotlib.pyplot as plt
import time 
# from transformers import (
# 	WEIGHTS_NAME,
# 	AdamW,
# 	get_linear_schedule_with_warmup,
# 	squad_convert_examples_to_features,
# )

torch.autograd.set_detect_anomaly(True)


def loss_REINFORCE( rewards, probabilities, device):
	loss = 0
	n = len(rewards)
	cumulative_reward = [0]
	for i in range(n):
		cumulative_reward.append(rewards[n - i - 1].squeeze() + cumulative_reward[-1])
		loss = loss + torch.log(probabilities[n - i - 1]) * cumulative_reward[-1]
	return loss.mean() / n


def batch_loss_REINFORCE(buffer, ids, rewards, probabilities, indices, device):
	loss, reward = 0, 0
	bsz = len(rewards)
	#print(bsz)
	for i in range(bsz): #for entire batch
		#actor loss
		actor_loss = loss_REINFORCE(rewards[i][:indices[i] + 1], probabilities[i][:indices[i] + 1], device).float()
		
		#calculating mapo loss
		
		#fetching value of the buffer at key=index
		index = str(ids[i])
		
		#bringing rewards, probabilities, indices in correct format
		mapo_rewards = []
		mapo_probabilities = []
		mapo_indices = []
		traj_prob_list = []

		l = len(buffer[index])
		for j in range(l):
			mapo_rewards.append((buffer[index][j]["rewards"]).tolist())
			mapo_probabilities.append(buffer[index][j]["probabilities"].tolist())
			mapo_indices.append(buffer[index][j]["indices"].item())
			
			#sum of all the probabilitie of the trajectories present in the buffer
			traj_prob_list.append(buffer[index][j]["traj_prob"].item()) 	 
		
		#since the probabilities were stored in log form, we take exponent of each and add them
		x = torch.exp(torch.tensor(traj_prob_list))
		
		#final sum of probabilitie of all the trajectories present 
		pi_b = torch.sum(x)
		
		mapo_rewards = torch.Tensor(mapo_rewards)
		mapo_probabilities = torch.Tensor(mapo_probabilities)
		mapo_probabilities =  mapo_probabilities.view(len(buffer[index]), -1, 3)
		mapo_indices = torch.Tensor(mapo_indices)
		mapo_indices = mapo_indices.int()
		
		#calculating avg loss of trajectories in the buffer for ith instance
		mapo_loss = 0 
		for j in range(len(buffer[index])):
			mapo_loss +=  loss_REINFORCE(mapo_rewards[j][:mapo_indices[j] + 1], mapo_probabilities[j][:mapo_indices[j] + 1], device).float()
		mapo_loss =  mapo_loss / len(mapo_rewards)
		
		#loss  of instance
		loss_for_instance = pi_b * mapo_loss + (1-pi_b) * actor_loss
		
		#loss of batch
		loss += loss_for_instance
		reward += rewards[i][indices[i]].float()
	loss = - loss / bsz
	return loss, reward


#def make_eq(eq_to_be_expanded):

# def expand_traj(traj,args):
# 	l = list(traj.keys())
# 	max_key = max(l)
# 	min_key = min(l)
# 	eq_to_be_expanded = np.array(traj[max_key])
# 	make_eq(eq_to_be_expanded)
	

# 	print(eq_to_be_expanded)
# 	return eq_to_be_expanded



def train(args, dataset, encoder, decoder, device, save_path="models/"):
	n_epochs = args.num_epochs
	encoder_optimizer = Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	encoder_scheduler = lr_scheduler.StepLR(encoder_optimizer, step_size=args.step_size, gamma=args.gamma)
	decoder_optimizer = Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	decoder_scheduler = lr_scheduler.StepLR(decoder_optimizer, step_size=args.step_size, gamma=args.gamma)
	model_save_path = save_path + "epoch.pt"

	print("In train.py")
	if args.mode == 'test':
		encoder.eval()
		decoder.eval()

	#realted to buffer
	#mapo_buffer= []

	#buffer is a dictionary {key: index, value : buffer for that index(list of dictionaries)}
	mapo_buffer = {} 
	k = 5 #top k traj i.e size of buffer would be 
	#top_k_traj = 100
	
	for i in range(n_epochs):
		print("epoch: ", i)
		start_time = time.time()
		total_reward = 0
		n_examples_processed = 0
		n_batch_processed = 0
		total_loss = 0

		for batch in tqdm(dataset):
			encoder_optimizer.zero_grad()
			decoder_optimizer.zero_grad()
			loss = 0
			ids, src_insts, src_feat_insts, src_lengths, num_insts, num_lengths, num_masks, ans_insts, _ = batch[-9:]
			#print(ids, src_insts, src_feat_insts, src_lengths, num_insts, num_lengths, num_masks, ans_insts)
			
			src_insts = src_insts.to(device)
			src_feat_insts = src_feat_insts.to(device)
			# src_lengths = src_lengths.to(device)
			# print("num_masks", num_masks.shape, num_masks)

			if args.encoder == 'gru':
				# print("Input to encoder")
				output, hidden = encoder(src_insts, src_lengths, src_feat_insts)
				# print("Encoder output")
				# print("shapes", output.shape, hidden.shape)
				# print("Content", output, hidden)

			batch_size = num_insts.shape[0]
			operand_dict = num_insts.to(device)
			probability_mask = num_masks.to(device)
			operand_dict = torch.cat((num_insts, torch.zeros(
				(operand_dict.shape[0], args.operand_vocab_size - operand_dict.shape[1])).double()), 1)
			probability_mask = torch.cat((num_masks, torch.zeros(
				(probability_mask.shape[0], args.operand_vocab_size - probability_mask.shape[1])).double()), 1)

			#eqntree_dict = [[Et(num.item()) for num in operand] for operand in operand_dict]

			operand_dict = operand_dict.to(device) #operand dict contains all the operands
			probability_mask = probability_mask.to(device)
			#print(operand_dict[1, :])

			rewards = torch.tensor([[]] * batch_size).to(device)
			probabilities = torch.tensor([[]] * batch_size).to(device)
			batch_size = len(num_insts)
			indices = torch.zeros(batch_size).fill_(-1).to(device)

			decoder_input = torch.tensor([[START_OP] * batch_size], device=device).T
			result = torch.tensor([0])
			

			#traj_dict = [dict() for x in range(batch_size)] #to store all the trajectroies as key: result op_id, value : op_id op1_id op2_id
			
			#how many time steps we want to decode (to give it maximum length)
			for j in range(args.max_decode_length):

				# print("decoder_input", decoder_input.shape)
				# print("hidden", hidden.shape)
				if args.mode == 'train': # getting distribution over operator and operand ids
					#op_id: the sampled operator
					operator, op1, op2, hidden, op_id = decoder(decoder_input, hidden.to(device), sample=True)
				elif args.mode == 'test':
					operator, op1, op2, hidden, op_id = decoder(decoder_input, hidden.to(device), sample=False)
				

				op1_prob = MaskedSoftmax()(op1, probability_mask) #masking the operands which are absent in the question
				op2_prob = MaskedSoftmax()(op2, probability_mask)

				# print("op1_prob", op1_prob.shape)
				# print("op2_prob", op2_prob.shape)

				if args.mode == 'train':
					#op1_id, op2_id sampled operands
					op1_id = torch.multinomial(op1_prob, 1).T.squeeze()
					op2_id = torch.multinomial(op2_prob, 1).T.squeeze()

				elif args.mode == 'test':
					op1_id = torch.argmax(op1_prob, 1)
					op2_id = torch.argmax(op2_prob, 1)
				
				# for b in range(batch_size):
				# 	eqtree_dict_b=eqntree_dict[b]
				# 	t = Et(OPERATOR_DICT[op_id[0][b].item() + 1])
				# 	print(t)
				# 	t.left = eqtree_dict_b[op1_id[b].item()]
				# 	t.right = eqtree_dict_b[op2_id[b].item()]
				# 	eqtree_dict_b[num_lengths[b]] = t
				# 	eqntree_dict[b]=eqtree_dict_b
				
			
				#storing the probability distributins at each step
				#all three probabilities at each step
				probability_stack = torch.stack([operator[range(batch_size), op_id.squeeze()], op1_prob[
						range(batch_size), op1_id], op2_prob[range(batch_size), op2_id]], dim=-1)
				#print(probability_stack[0])
				probabilities = torch.cat([probabilities, probability_stack], dim=-1)
				result = execute_supervised(op_id, op1_id, op2_id, operand_dict)
				
				#calculating and storing the reward(1 or 0)
				rewards = torch.cat([rewards, (
							torch.abs(result - torch.tensor(ans_insts).double().to(device)) < 1e-3).float().unsqueeze(
					-1)], dim=-1)
		
				#when to stop(after finding the correct answer)	
				indices[(op_id.squeeze() == 0) & (indices == -1)] = j
				if j == 0: rewards[(op_id.squeeze() == 0), j] = 0
				indices[(rewards[range(batch_size), -1] == 1) & (indices == -1)] = j
			
				#adding the new number to the operand dict
				
				#print("op_before",operand_dict[1, : 5])
				#print(num_lengths[1])
				operand_dict[range(len(operand_dict)), num_lengths] = result.squeeze().detach()
				#print("op_after",operand_dict[1, : 5])

				#to keep track of trajectories
				#trajectroies as key: result_op_id, value : op_id op1_id op2_id
				# op_id = op_id.cpu().squeeze()
				# op1_id = op1_id.cpu().squeeze()
				# op2_id = op2_id.cpu().squeeze()
				# for b in range(batch_size):
				# 	# print(num_lengths[b].item())
				# 	# print(op_id[b].item(), op1_id[b].item(), op2_id[b].item())
				# 	traj_dict[b][num_lengths[b].item()] = [op_id[b].item(), op1_id[b].item(), op2_id[b].item()]
				# op_id = op_id.to(device)
				# op1_id = op1_id.to(device)
				# op2_id = op2_id.to(device)

				vocab_size_add = op_id.clone().detach().cpu()
				vocab_size_add = torch.where(vocab_size_add > 0, torch.tensor(1), vocab_size_add).squeeze().long()
				num_lengths += vocab_size_add
				
				probability_mask_new = probability_mask.clone()
				probability_mask_new[range(len(num_lengths)), num_lengths - 1] = 1
				probability_mask = probability_mask_new.clone()
				decoder_input = op_id.clone().detach().T

		
			#where i am stoppped	
			indices[(indices == -1)] = (args.max_decode_length - 1)
			reward = rewards[range(batch_size), indices.long()].sum().item()
			
			#converting reward to +1 &-1 from 0 & 1
			# print("prev rewards", rewards[0])
			for i in range(len(rewards)):
				for j in range(len(rewards[i])):
					rewards[i][j] = 2 * rewards[i][j] - 1
			#exit()
			#rewards[list(range(batch_size)), indices.long()] = 2 * rewards[list(range(batch_size)), indices.long()] - 1


			# compute and print final expression trees
			# print(len(eqntree_dict))
			# final_predicted_trees = [eqntree_dict[b][indices.long()[b]] for b in range(batch_size)]
			# print("predicted_trees")
			# for b in range(batch_size):
			# 	print(final_predicted_trees[b].print_eqn(final_predicted_trees[b]))
			
			#put these trajectories into the buffer
			#buffer is maintained in ascending order in terms of probabilities
			
			
			# print("adjusting buffer")

			#*********************maintaing buffer*************************#
			#for every index(instance),store top 5 trajectories in its buffer

			index_to_instance =list(ids)
			for i, id in enumerate(index_to_instance):			#all the index in a batch
				id = str(id)
				#calculating log probability of the trajectory of ith instance of the buffer
				traj_prob = int(0) 
				for x in range(3* args.max_decode_length):
					traj_prob = traj_prob + torch.log(probabilities[i][x])

				if id in mapo_buffer.keys():
					if len(mapo_buffer[id]) < k:
						mapo_buffer[id].append({"rewards": rewards[i], "probabilities": probabilities[i],
						"indices": indices[i], "traj_prob" : traj_prob})
					else:
						#print("sorting..", btch_len)
						mapo_buffer[id] = sorted(mapo_buffer[id], key=lambda y: y["traj_prob"])
						if mapo_buffer[id][0]["traj_prob"] < traj_prob: #do it on the basis of rewards(add up all the rewards in the decoding steps)
							mapo_buffer[id][0] = {"rewards": rewards[i], "probabilities": probabilities[i],
							"indices": indices[i], "traj_prob" : traj_prob}
				
				else:
					mapo_buffer[id] = [{"rewards": rewards[i], "probabilities": probabilities[i],
						"indices": indices[i], "traj_prob" : traj_prob}]
			#print(mapo_buffer["318"])
			
						
			loss, _ = batch_loss_REINFORCE(mapo_buffer, ids, rewards, probabilities.view(batch_size, -1, 3), indices.int(), device)
		
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
		
		l = l.append(total_loss / n_batch_processed)
		checkpoint = {
			'epoch': i,
			'loss': total_loss / n_batch_processed,
			'encoder_state_dict': encoder.state_dict(),
			'decoder_state_dict': decoder.state_dict(),
		}
		if (i%args.save_freq) == 0:
			model_save_path = save_path + "epoch - " + str(i) + ".pt"
			torch.save(checkpoint, model_save_path)
		end_time = time.time()
		print("elapsed time for a epoch", (end_time - start_time)/60)
	plt.plot(l)