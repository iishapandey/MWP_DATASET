import argparse, os
import torch
from train import train
from train_kb import train_kb
from train_stop import train_stop
from train_with_beamsearch import train_with_beamsearch
from train_with_beamsearch_tensorized_final import train as train_with_beamsearch
from train_with_beamsearch_kb import train_with_beamsearch_kb
from train_with_beamsearch_stop import train_with_beamsearch_stop
from test_with_beamsearch import test_with_beamsearch
from test_with_beamsearch_stop import test_with_beamsearch_stop
from test import test_with_beamsearch as test
from gru_encoder import GruEncoder
from data_loader import MWP_Gru_Dataset, collate_fn_gru
from DecoderOperatorOperandSelector import DecoderOperatorOperandSelector
from DecoderFCN1 import DecoderFCN1
from DecoderFCN2 import DecoderFCN2
from data_preprocess import Et
from equation_tac import EquationTAC
import matplotlib.pyplot as plt
from unit_kb import Unit
import random

#device = torch.device("cuda:2")
device = torch.device("cuda")
# random.seed(0)
# torch.manual_seed(0)


def build_model(args):
    global device
    #if not args.no_cuda:
    #    device = "cuda:2"
    #else:
    #    device = "cpu"
    print(device)
    if args.encoder == 'gru':
        encoder = GruEncoder(args.hidden_size, args.embed_size, args.feat_size, args.vocab_size, bidirectional=args.bidirectional, 
                             n_layers=args.n_layers, dropout=args.dropout)
    # elif args.encoder == 'bert':
    #     encoder = BertEncoder(args,'bert-base-chinese')
    
    if args.decoder == 'gru':
        decoder = DecoderOperatorOperandSelector(args.hidden_size, args.operator_vocab_size - 1, args.operand_vocab_size,
                                             device, n_layers=args.n_layers, dropout=args.dropout)
    elif args.decoder == 'fcn1':
        decoder = DecoderFCN1(args.embed_size, args.hidden_size, args.operator_vocab_size - 1, args.operand_vocab_size, 
                                             device, dropout=args.dropout)
    
    elif args.decoder == 'fcn2':
        decoder = DecoderFCN2(args.embed_size, args.hidden_size, args.operator_vocab_size - 1, args.operand_vocab_size, 
                                             device, dropout=args.dropout)
    
    return encoder, decoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Math word problem solver')

    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-hidden_size', type=int, default=512)
    parser.add_argument('-embed_size', type=int, default=128)
    parser.add_argument('-feat_size', type=int, default=4)
    parser.add_argument('-bidirectional', action="store_true", default=True)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-operator_vocab_size', type=int, default=7)
    parser.add_argument('-operand_vocab_size', type=int, default=40)
    parser.add_argument('-vocab_size', type=int)
    parser.add_argument('-num_epochs', type=int, default=204)
    parser.add_argument('-max_decode_length', type=int, default=15)
    parser.add_argument('-no_cuda', action="store", dest="no_cuda")
    parser.add_argument('-encoder', type=str, default='gru')
    parser.add_argument('-decoder', type=str, default='fcn1')
    parser.add_argument('-train_data_path', type=str, default="data/math23k/gru/train.pt")
    parser.add_argument('-test_data_path', type=str, default="data/math23k/gru/train.pt")
    parser.add_argument('-weight_decay', type=float, default=1e-5)
    parser.add_argument('-step_size', type=int, default=75)
    parser.add_argument('-gamma', type=float, default=0.7)
    parser.add_argument('-warmup', type=int, default=0)
    parser.add_argument('-adam_epsilon', type=float, default=1e-8)
    parser.add_argument('-mode', type=str, default='train')
    parser.add_argument('-model_path', type=str, default="math23k/")
    parser.add_argument('-save_freq', type=int, default=2)
    parser.add_argument('-save_path', type=str, default="math23k/")
    parser.add_argument('-use_beam_search', type=bool, default=False)
    parser.add_argument('-beam_width', type=int, default=5)
    parser.add_argument('-use_units', type=bool, default=False)
    parser.add_argument('-use_stop', type=bool, default=False)
    parser.add_argument('-print_eqn', type=bool, default=True)

    args = parser.parse_args()

    if args.mode == 'train' : 
        data = torch.load(args.train_data_path)
    elif args.mode == 'test': 
        data = torch.load(args.test_data_path)
    '''


    if args.encoder == 'gru':
        dataset = MWP_Gru_Dataset(ids=data['dict']['ids'], word2idx=data['dict']['vocab'], feat2idx={},
                            src_insts=data['dict']['src'], src_feat_insts=data['dict']['feat'],
                            num_insts=data['dict']['nums'], num_unit_insts=data['dict']['num_units'],
                            num_rate_insts=data['dict']['num_rates'], ans=data['dict']['ans'], 
                            ans_units=data['dict']['ans_units'], ans_rates=data['dict']['ans_rates'], 
                            eqns=data['dict']['eqns'])
    '''

#creating torch dataset from the pt file
    if args.encoder == 'gru':
        dataset = MWP_Gru_Dataset(ids=data['dict']['ids'], word2idx=data['dict']['vocab'], feat2idx={},src_insts=data['dict']['src'],
                            src_feat_insts=data['dict']['feat'], num_insts=data['dict']['nums'], ans=data['dict']['ans'], eqns=data['dict']['eqns'])

       
        
        print("In main.py")
        # print('word2idx=', data['dict']['vocab'])  # word2ind
       	# print('src_insts=', data['dict']['src'][0])  # input sentence in the form of list of vocab word indices
        # print('src_feat_insts=', data['dict']['feat'][0]) # list where value is 1 if word is number, else 0
        # print('num_insts=', data['dict']['nums'][0]) # list of numbers in question appended with [1.0, 3.14]
        # print('num_unit_insts=', data['dict']['num_units'][0]) # list of units of the above numbers
        # print('num_rate_insts=', data['dict']['num_rates'][0]) # list where value is true if the above unit is rate of something
        # print('ans=', data['dict']['ans'][0]) # numerical ans
        # print('ans_units=', data['dict']['ans_units'][0]) # unit of ans
        # print('ans_rates=', data['dict']['ans_rates'][0]) # whether ans is rate
        # print('eqns=', data['dict']['eqns'][0]) # numerical eqn with numbers (infix)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn_gru)

    # if args.encoder == 'bert':
    #     dataset = MWP_Bert_Dataset(input_ids=data['dict']['input_ids'], token_type_ids=data['dict']['token_type_ids'],
    #                         attention_mask=data['dict']['attention_mask'], ids=data['dict']['ids'], word2idx=data['dict']['vocab'], 
    #                         feat2idx={}, src_insts=data['dict']['src'], src_feat_insts=data['dict']['feat'],
    #                         num_insts=data['dict']['nums'], num_unit_insts=data['dict']['num_units'],
    #                         num_rate_insts=data['dict']['num_rates'], ans=data['dict']['ans'], 
    #                         ans_units=data['dict']['ans_units'], ans_rates=data['dict']['ans_rates'], 
    #                         eqns=data['dict']['eqns'])
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn_bert)
    
    args.vocab_size = dataset.src_vocab_size + 2  # +2 -> pad,unk
    encoder, decoder = build_model(args)

    encoder.to(device)
    decoder.to(device)

    '''
    checkpoint = torch.load("math23k_tensorized_final/epoch - 120.pt", map_location='cpu') # 100: 83.16, 112:
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    acc = test_with_beamsearch(args, dataloader, encoder,  decoder, device, args.beam_width)
    # test(args, dataloader, encoder,  decoder, device)
    exit()
    # print(args)
    '''
    if not args.use_beam_search:
        if args.mode == 'train':
            if args.use_stop: #this if is not used
                train_stop(args, dataloader, encoder,  decoder, device, args.save_path)
            elif args.use_units:
                train_kb(args, dataloader, encoder,  decoder, device, args.save_path)
            else:
                train(args, dataloader, encoder,  decoder, device, args.save_path) #this is being actually called for training
        elif args.mode == 'test': #the model which we passed
            if args.model_path.endswith('.pt'):
                checkpoint = torch.load(args.model_path)
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
                acc = train(args, dataloader, encoder,  decoder, device)
                print("Acc - {}".format(acc))
            else: #testing for all the models(all cehckpoint models)
                acc_arr = []
                for i in range(0, args.num_epochs, args.save_freq):
                    if not os.path.isfile(args.model_path + "epoch - {}.pt".format(i)):
                        continue
                    checkpoint = torch.load(args.model_path + "epoch - {}.pt".format(i))
                    encoder.load_state_dict(checkpoint['encoder_state_dict'])
                    decoder.load_state_dict(checkpoint['decoder_state_dict'])
                    acc = train(args, dataloader, encoder,  decoder, device, args.save_path)
                    print("Epoch {} Acc - {}".format(i, acc))
                    acc_arr.append(acc)
                print("Best Acc - {}".format(max(acc_arr)))
                plt.plot(acc_arr)
                plt.show()
    elif args.mode == 'train':
        if args.use_stop:
            train_with_beamsearch_stop(args, dataloader, encoder, decoder, device, args.save_path, args.beam_width)
        elif args.use_units:
            train_with_beamsearch_kb(args, dataloader, encoder, decoder, device, args.save_path, args.beam_width)
        else: #training here
            train_with_beamsearch(args, dataloader, encoder, decoder, device, args.save_path, args.beam_width)
    elif args.mode == 'test':
        if args.model_path.endswith('.pt'):
            checkpoint = torch.load(args.model_path)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            if args.use_stop:
                acc = test_with_beamsearch_stop(args, dataloader, encoder,  decoder, device, args.beam_width)
            else:
                acc = test_with_beamsearch(args, dataloader, encoder,  decoder, device, args.beam_width)
            print("Acc - {}".format(acc))
        else:
            acc_arr = []
            for i in range(0, args.num_epochs, args.save_freq):
                if not os.path.isfile(args.model_path + "epoch - {}.pt".format(i)):
                    continue
                checkpoint = torch.load(args.model_path + "epoch - {}.pt".format(i))
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
                if args.use_stop:
                    acc = test_with_beamsearch_stop(args, dataloader, encoder,  decoder, device, args.beam_width)
                else:
                    acc = test_with_beamsearch(args, dataloader, encoder,  decoder, device, args.beam_width)
                print("Epoch {} Acc - {}".format(i, acc))
                acc_arr.append(acc)
            print("Best Acc - {}".format(max(acc_arr)))
            plt.plot(acc_arr)
            plt.show()
