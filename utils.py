"""
Utility functions:
-> pad questions
-> parse tensorboard event logs and save into csvs
-> generate statistics of length of questions
-> plot training and validation statistics
-> plot vqa accuracy
-> plot all accuracies
-> predict answers given an image and questions in that image
"""
from models.baseline import VQABaseline
from PIL import Image
from tensorboard.backend.event_processing import event_accumulator

import collections
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd
import string
import torch
import torch.nn as nn
import torchvision.transforms as transforms

def preprocess_text(text):
    """
        Converts a string to lower case, removes punctuations.
    """
    text_token_list = text.strip().split(',')
    text  = ' '.join(text_token_list)

    # Remove punctuations
    table = str.maketrans('', '', string.punctuation)
    words = text.strip().split()
    words = [w.translate(table) for w in words]

    # Set to lowercase & drop empty strings
    words = [word.lower() for word in words if word != '' and word != 's']
    
    text  = ' '.join(words)
    return text

def pad_sequences(l, max_length):
    """
        Pad question with <pad> token (i.e., idx 0) till max_length, if
        question length exceeds max_length cut it to max_length
    """
    padded = np.zeros((max_length,), np.int64)
    if len(l) > max_length:
        padded[:] = l[:max_length]
    else:
        padded[:len(l)] = l
    return padded

def parse_tb_logs(log_directory, run_name, epoch_or_step = 'epoch'):
    """
        Given log directory and the expirement run name, gather the tensorboard
        events using event accumulator, extract the train and val statistics at
        either epoch or step level and saves them into a csv in the same log directory
    """
    if epoch_or_step == 'step':
        run_name   += '_step'

    train_losses      = []
    train_accuracies  = []
    val_losses        = []
    val_accuracies    = []

    directory         = os.path.join(log_directory, run_name)
    for filename in os.listdir(directory):
        ea = event_accumulator.EventAccumulator(os.path.join(directory, filename),
                                                size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        train_losses     += ea.Scalars('Train_Loss')
        train_accuracies += ea.Scalars('Train_Accuracy')
        if epoch_or_step == 'epoch':
            val_losses       += ea.Scalars('Val_Loss')
            val_accuracies   += ea.Scalars('Val_Accuracy')

    train_losses          = pd.DataFrame(train_losses).rename(columns = {'value': 'train_loss'})
    train_accuracies      = pd.DataFrame(train_accuracies).rename(columns = {'value': 'train_accuracy'})
    if epoch_or_step == 'epoch':
        val_losses            = pd.DataFrame(val_losses).rename(columns = {'value': 'val_loss'})
        val_accuracies        = pd.DataFrame(val_accuracies).rename(columns = {'value': 'val_accuracy'})

    df                    = train_losses
    df['train_accuracy']  = train_accuracies['train_accuracy']
    if epoch_or_step == 'epoch':
        df['val_loss']        = val_losses['val_loss']
        df['val_accuracy']    = val_accuracies['val_accuracy']
        df                    = df.rename(columns = {'step': 'epoch'})

    df = df.sort_values(by = 'wall_time', ascending = True)
    df = df.drop_duplicates(['epoch'] if epoch_or_step == 'epoch' else ['step'], keep = 'last')

    df.to_csv(os.path.join(log_directory, run_name + '.csv'), index = False)

def get_question_length_stats(data_directory):
    """
        Reads the preprocessed train data and computes the statistics of
        length of questions (i.e., number of words) and their frequencies.
    """
    with open(os.path.join(data_directory, 'train_data.txt'), 'r') as f:
        train_data = f.read().strip().split('\n')

    questions      = [x.split('\t')[1].strip() for x in train_data]
    words          = [x.split() for x in questions]
    lengths        = [len(x) for x in words]

    count          = collections.Counter(lengths)
    plt.bar(count.keys(), count.values())
    plt.xlabel("Sequence Length")
    plt.ylabel("Number of Questions")
    count          = sorted(count.items())

    df             = pd.DataFrame(count, columns=['sequence_length', 'count'])
    df['perc']     = 100 * df['count'].cumsum() / df['count'].sum()

    return df

def plot_train_val_stats(log_directory, run_name, epoch_or_step = 'epoch'):
    """
        Given log directory and the expirement run name. plots the train loss, accuracy
        and validation loss, accuracy by reading the csv files parse_tb_logs() generates
    """
    if epoch_or_step == 'step':
        run_name   += '_step'
    df              = pd.read_csv(os.path.join(log_directory, run_name + '.csv'))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    # Loss plot
    plt.sca(axs[0])
    plt.plot(df[epoch_or_step].values, df["train_loss"].values, label = "Train Loss")
    if epoch_or_step == 'epoch':
        plt.plot(df[epoch_or_step].values, df["val_loss"].values, label = "Val Loss")
    plt.xlabel("Number of Epochs" if epoch_or_step == 'epoch' else "Number of Steps")
    plt.ylabel("Loss")
    plt.title("Loss vs Number of Epochs" if epoch_or_step == 'epoch' else "Number of Steps")
    plt.legend()

    # Accuracy plot
    plt.sca(axs[1])
    plt.plot(df[epoch_or_step].values, df["train_accuracy"].values, label = "Train Accuracy")
    if epoch_or_step == 'epoch':
        plt.plot(df[epoch_or_step].values, df["val_accuracy"].values, label = "Val Accuracy")
    plt.xlabel("Number of Epochs" if epoch_or_step == 'epoch' else "Number of Steps")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Epochs" if epoch_or_step == 'epoch' else "Number of Steps")
    plt.legend()
    plt.show()

def plot_train_accuracies(log_directory, run_names):
    for run_name in run_names:
        df              = pd.read_csv(os.path.join(log_directory, run_name + '.csv'))
        plt.plot(df['epoch'].values, df["train_accuracy"].values, label = run_name)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Train Accuracy")
    plt.title("Train Accuracy vs Number of Epochs")
    plt.legend()
    plt.show()

def plot_val_accuracies(log_directory, run_names):
    for run_name in run_names:
        df              = pd.read_csv(os.path.join(log_directory, run_name + '.csv'))
        plt.plot(df['epoch'].values, df["val_accuracy"].values, label = run_name)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Val Accuracy")
    plt.title("Val Accuracy vs Number of Epochs")
    plt.legend()
    plt.show()

def plot_all_accuracies(log_directory, run_names):
    """
        plots separate plots of train, val and vqa accuracy for all the run_names together
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    plt.sca(axs[0])
    for run_name in run_names:
        df              = pd.read_csv(os.path.join(log_directory, run_name + '.csv'))
        plt.plot(df['epoch'].values, df["train_accuracy"].values, label = run_name)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Train Accuracy")
    plt.title("Train Accuracy vs Number of Epochs")
    plt.legend()
    
    plt.sca(axs[1])
    for run_name in run_names:
        df              = pd.read_csv(os.path.join(log_directory, run_name + '.csv'))
        plt.plot(df['epoch'].values, df["val_accuracy"].values, label = run_name)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Val Accuracy")
    plt.title("Val Accuracy vs Number of Epochs")
    plt.legend()
    
   

def get_model(model_type, vocab_size, use_image_embedding, use_dropout, output_size, image_model_type, attention_mechanism, word_embedding_size, lstm_state_size, bi_directional = False, max_length = 14, use_glove = False, use_lstm = True, embedding_file_path = None, use_clip = False):
    """
        Instantiates the pytorch model given the appropriate parameters.
    """
    model = None

    if model_type == 'baseline':
        model = VQABaseline(vocab_size = vocab_size, use_image_embedding = use_image_embedding, use_dropout = use_dropout,
                            output_size = output_size, image_model_type = image_model_type, attention_mechanism = attention_mechanism,
                            word_embedding_size = word_embedding_size, lstm_hidden_size = lstm_state_size, bi_directional = bi_directional, max_seq_len = max_length, use_glove = use_glove, use_lstm = use_lstm, embedding_file_path = embedding_file_path, use_clip = use_clip)
    else:
        raise Exception(f'Model Type {model_type} is not supported')

    return model

def get_image_path(data_dir, image_id, mode = 'test'):
    """
        returns the path for an image given the data directory and the mode
    """
    image_path = ""
    if mode == 'train' or mode == 'val':
        image_path = f"{data_dir}/images/{mode}2014/COCO_{mode}2014_{int(image_id):012d}.jpg"
    else:
        image_path = f"{data_dir}/images/{mode}2015/COCO_{mode}2015_{int(image_id):012d}.jpg"
    
    return image_path

def get_image_to_questions(data_dir, mode = 'test'):
    """
        given a mode and data directory, returns a mapping from image id to the questions for that image
    """
    if mode == 'train' or mode == 'val':
        ques_file = f"v2_OpenEnded_mscoco_{mode}2014_questions.json"
    else:
        ques_file = f"v2_OpenEnded_mscoco_{mode}2015_questions.json"
    questions     = json.load(open(os.path.join(data_dir, 'questions', ques_file), 'r'))['questions']
    image_ids     = set(q["image_id"] for q in questions)
    imageToQ      = {image_id: [] for image_id in image_ids}
    for q in questions:
        imageToQ[q["image_id"]].append(q["question"])

    return imageToQ

def answer_these_questions(data_dir, model_dir, image_path, questions, model_type = 'baseline', run_name = 'baseline_512',
                           top_k = 1000, max_length = 14, image_model_type = 'vgg16', word_embedding_size = 300, use_dropout = True,
                           lstm_hidden_size = 512, attention_mechanism = 'element_wise_product', num_answers = 5,
                           bi_directional = False, use_glove = False, use_lstm = True, embedding_file_name = 'word_embeddings_glove.pkl',model=None):
    """
        prints the predicted answers given an image and the questions for that image.
        - data_dir:            directory of images and preprocessed data
        - model_dir:           directory where the saved models are present
        - image_path:          image path
        - questions:           questions for the image
        - model_type:          VQA model choice
        - run_name:            which experiment to use
        - top_k:               select top_k frequent answers that was used during training for this run_name
        - max_length:          max number of words in the question that was used during training for this run_name
        - image_model_type:    type of CNN for the Image Encoder used for the experiment
        - word_embedding_size: word embedding size used during training
        - lstm_hidde_size:     lstm hidden state size used during training
        - attention_mechanism: attention mechanism used during training
        - num_answers:         return num_answers for each question
        - bi_directional:      whether the lstm trained on was bi directional
        - use_lstm:            True if lstm is used
        - use_glove:           True if glove embeddings are used
        - embedding_file_name: glove embeddings path file name
    """
    n          = len(questions)
    transform  = transforms.Compose([
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    orig_image = Image.open(image_path)
    img        = transform(orig_image.convert('RGB')).repeat(n, 1, 1, 1)

    labelfreq  = pickle.load(open(os.path.join(data_dir, f'answers_freqs.pkl'), 'rb'))
    label2idx  = {x[0]: i+1 for i, x in enumerate(collections.Counter(labelfreq).most_common(n = top_k - 1))}
    label2idx["<unk>"]  = 0
    idx2label  = {i:l for l, i in label2idx.items()}

    word2idx   = pickle.load(open(os.path.join(data_dir, 'questions_vocab.pkl'), 'rb'))["word2idx"]

    orig_ques  = questions
    questions  = [[word2idx[w] if w in word2idx else word2idx['<unk>'] for w in preprocess_text(question).split()] for question in questions]
    questions  = [pad_sequences(question, max_length) for question in questions]
    questions  = torch.from_numpy(np.array(questions))
    
    # make the predictions
    preds = model(img, questions)
    preds = torch.softmax(preds, 1)

    probs, indices = torch.topk(preds, k = num_answers, dim = 1)
    probs          = probs.tolist()
    indices        = indices.tolist()

    orig_image.show()
    for i, question in enumerate(orig_ques):
        print(question)
        for index, prob in zip(indices[i], probs[i]):
            if index != 0:
                print(idx2label[index])
        print("\n")
