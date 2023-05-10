from dataset import VQADataset
from models.baseline import VQABaseline
from PIL import Image
from torch.optim import Adadelta, Adam, lr_scheduler, RMSprop
from torch.utils.data import Dataset, DataLoader
from train import *
from utils import *

import argparse
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms

def boolstr(s):
    """
        Argument parser for boolean string.
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main():
    parser = argparse.ArgumentParser(description='VQA')

    parser.add_argument('--data_dir',               type=str,       help='directory of the preprocesses data', default='/home/an3729/datasets')
    parser.add_argument('--model_dir',              type=str,       help='directory to store model checkpoints (saved as run_name.pth)', default='/home/an3729/checkpoints')
    parser.add_argument('--log_dir',                type=str,       help='directory to store log files (used to generate run_name.csv files for training results)', default='/home/an3729/logs')
    parser.add_argument('--run_name',               type=str,       help='unique experiment name (used as prefix for all data saved on a run)', default='testrun', required=True)
    parser.add_argument('--model',                  type=str,       help='VQA model choice', choices=['baseline'], default='baseline', required=True)
    parser.add_argument('--image_model_type',       type=str,       help='Type of CNN for the Image Encoder', choices=['vgg16', 'resnet152'], default='vgg16')

    parser.add_argument('--use_image_embedding',    type=boolstr,   help='Use precomputed embeddings directly', default=True)
    parser.add_argument('--top_k_answers',          type=int,       help='Top K answers used to train the model (output classifier size)', default=1000)
    parser.add_argument('--max_length',             type=int,       help='max sequence length of questions', default=14) # covers 99.7% of questions
    parser.add_argument('--word_embedding_size',    type=int,       help='Word embedding size for the embedding layer', default=300)
    parser.add_argument('--lstm_state_size',        type=int,       help='LSTM hidden state size', default=512)

    parser.add_argument('--batch_size',             type=int,       help='batch size', default=512)
    parser.add_argument('--epochs',                 type=int,       help='number of epochs i.e., final epoch number', default=50)
    parser.add_argument('--learning_rate',          type=float,     help='initial learning rate', default=1.0)
    parser.add_argument('--optimizer',              type=str,       help='choice of optimizer', choices=['adam', 'adadelta'], default='adadelta')
    parser.add_argument('--use_dropout',            type=boolstr,   help='use dropout', default=True)
    parser.add_argument('--use_sigmoid',            type=boolstr,   help='use sigmoid activation to compute binary cross entropy loss', default=False)
    parser.add_argument('--use_sftmx_multiple_ans', type=boolstr,   help='use softmax activation with multiple possible answers to compute the loss', default=False)
    parser.add_argument('--ignore_unknowns',        type=boolstr,   help='Ignore unknowns from the true labels in case of use_sigmoid or use_sftmx_multiple_ans', default=True)
    parser.add_argument('--use_softscore',          type=boolstr,   help='use soft score for the answers, only applicable for sigmoid or softmax with multiple answers case', default=True)

    parser.add_argument('--print_stats',            type=boolstr,   help='flag to print statistics i.e., the verbose flag', default=True)
    parser.add_argument('--print_epoch_freq',       type=int,       help='epoch frequency to print stats at', default=1)
    parser.add_argument('--print_step_freq',        type=int,       help='step frequency to print stats at', default=300)
    parser.add_argument('--save_best_state',        type=boolstr,   help='flag to save best model, used to resume training from the epoch of the best state', default=True)
    parser.add_argument('--attention_mechanism',    type=str,       help='method of combining image and text embeddings', choices=['element_wise_product', 'sum', 'concat'], default='element_wise_product')
    parser.add_argument('--bi_directional',         type=boolstr,   help='True if lstm is to be bi-directional', choices=[True, False], default=False)
    parser.add_argument('--use_lstm',               type=boolstr,   help='True if lstm is to be used', choices=[True, False], default=True)
    parser.add_argument('--use_glove',              type=boolstr,   help='True if glove embeddings are to be used', choices=[True, False], default=False)
    parser.add_argument('--embedding_file_name',    type=str,       help='glove embedding path file', default='word_embeddings_glove.pkl')

    parser.add_argument('--random_seed',            type=int,       help='random seed for the experiment', default=43)
    parser.add_argument('--run_mode',            type=str,       help='train or test', default='train')
    parser.add_argument('--use_clip',            type=str,       help='use clip for question embedding', default=False)


    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform    = transforms.Compose([
                       transforms.Resize((224, 224)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # Create train and val datasets
    if args.model == 'baseline':
        train_ds     = VQADataset(args.data_dir, top_k = args.top_k_answers, max_length = args.max_length, transform = transform,
                                  use_image_embedding = args.use_image_embedding, image_model_type = args.image_model_type,
                                  ignore_unknowns = args.ignore_unknowns, use_softscore = args.use_softscore)
        val_ds       = VQADataset(args.data_dir, mode = 'val', top_k = args.top_k_answers, max_length = args.max_length, transform = transform,
                                  use_image_embedding = args.use_image_embedding, image_model_type = args.image_model_type,
                                  ignore_unknowns = args.ignore_unknowns, use_softscore = args.use_softscore)
        test_ds       = VQADataset(args.data_dir, mode = 'test', top_k = args.top_k_answers, max_length = args.max_length, transform = transform,
                                  use_image_embedding = args.use_image_embedding, image_model_type = args.image_model_type,
                                  ignore_unknowns = args.ignore_unknowns, use_softscore = args.use_softscore)
    else:
        raise Exception(f'Model Type {args.model} is not supported')

    num_gpus     = torch.cuda.device_count()
    batch_size   = args.batch_size
    # Get train and val data loaders
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
    val_loader   = DataLoader(val_ds, batch_size = batch_size, num_workers = 2, pin_memory = True)
    test_loader   = DataLoader(test_ds, batch_size = batch_size, num_workers = 2, pin_memory = True)

    # Initialize the model on the device, and also use dataparallel if num_gpus available is > 1.
    vocab_size   = len(pickle.load(open(os.path.join(args.data_dir, 'questions_vocab.pkl'), 'rb'))["word2idx"])
    model        = get_model(args.model, vocab_size, args.use_image_embedding, args.use_dropout, args.top_k_answers, args.image_model_type, args.attention_mechanism, args.word_embedding_size, args.lstm_state_size, args.bi_directional, args.max_length, args.use_glove, args.use_lstm, os.path.join(args.data_dir,args.embedding_file_name), args.use_clip)
    model        = nn.DataParallel(model).to(device) if num_gpus > 1 else model.to(device)
    
    # Optimizer - Adam/Adadelta
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr = args.learning_rate)
    else:
        optimizer = Adadelta(model.parameters(), lr = args.learning_rate)

    # Loss function
    if args.use_sigmoid:
        loss_fn   = nn.BCEWithLogitsLoss()
    elif args.use_sftmx_multiple_ans: # Allowing multiple answers in softmax
        loss_fn   = nn.LogSoftmax(dim = 1)
    else:
        loss_fn   =  nn.CrossEntropyLoss()

    if args.run_mode == 'train':

        # Train the model
        model, optim, best_accuracy = \
            train_model(model, train_loader, val_loader, loss_fn, optimizer, device,
                    args.model_dir, args.log_dir, epochs = args.epochs,
                    run_name = args.run_name, use_sigmoid = args.use_sigmoid, use_sftmx_multiple_ans = args.use_sftmx_multiple_ans,
                    save_best_state = args.save_best_state, print_stats = args.print_stats,
                    print_epoch_freq = args.print_epoch_freq, print_step_freq = args.print_step_freq)

        # Parse the log files and save epoch level and step level training stats in csv files.
        parse_tb_logs(args.log_dir, args.run_name, 'epoch')
        parse_tb_logs(args.log_dir, args.run_name, 'step')

    else:
        # load the model
        model.load_state_dict(torch.load(os.path.join(args.model_dir, args.run_name + '_best.pth'), map_location=torch.device('cpu')))
        model.eval()
        #model, loss, acc = test(model, test_loader, loss_fn, device)
        #print("Test loss = "+str(loss)+ " test accuracy = "+str(acc))
        answer_these_questions('/home/an3729/datasets', '/home/an3729/checkpoints', '/home/an3729/datasets/test/a.jpg', ['What is the projection of this image?'], model_type = 'baseline', run_name = 'vqa_384_blstm_baseline_testf',
                           top_k = 218,  use_dropout = True, num_answers = 5, bi_directional = False,  use_lstm = True, model=model)

if __name__ == '__main__':
    main()

# python main.py --run_name testrun --model baseline --data_dir ../Dataset --model_dir ../checkpoints --log_dir ../logs --epochs 1 --use_dropout True --use_sftmx_multiple_ans True --use_softscore False --word_embedding_size 500
