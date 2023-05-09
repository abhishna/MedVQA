from PIL import Image
from torch.utils.data import Dataset
from utils import pad_sequences

import collections
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as transforms

class VQADataset(Dataset):
    
    def __init__(self, data_dir, transform = None, mode = 'train', use_image_embedding = True, image_model_type = 'vgg16', top_k = 1000, max_length = 14, ignore_unknowns = True, use_softscore = True):
        """
            - data_dir:            directory of images and preprocessed data
            - transform:           any transformations to be applied to image (if not using embeddings)
            - mode:                train/val
            - use_image_embedding: use image embeddings directly that are stored using vectorize_images.py
            - top_k:               select top_k frequent answers for training
            - max_length:          max number of words in the question to use while training
            - ignore_unknowns:     while using multiple answers, don't include the unknowns as true answers
            - use_softscore:       for multiple answers, when set to True, uses weights based on the vqa metric of min(1, freq/3), else uses (freq/10)
        """
        self.data_dir              = data_dir
        self.transform             = transform
        self.mode                  = mode
        self.use_image_embedding   = use_image_embedding
        self.image_model_type      = image_model_type
        
        self.labelfreq             = pickle.load(open(os.path.join(data_dir, f'answers_freqs.pkl'), 'rb'))
        self.label2idx             = {x[0]: i+1 for i, x in enumerate(collections.Counter(self.labelfreq).most_common(n = top_k - 1))}
        self.label2idx["<unk>"]    = 0

        self.word2idx              = pickle.load(open(os.path.join(data_dir, 'questions_vocab.pkl'), 'rb'))["word2idx"]
        self.max_length            = max_length
        self.ignore_unknowns       = ignore_unknowns
        self.use_softscore         = use_softscore
        
        self.data_file             = f'{mode}_data.txt'

        # Read the processed data file
        with open(os.path.join(data_dir, self.data_file), 'r') as f:
            self.data              = f.read().strip().split('\n')
        
        self.image_features        = None
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
            returns image (or image embeddings), question (in vector form), answer, all_answers (10 answers), ans_score (a weight for each correct answer, used when allowed to train on multiple answers)
        """
        
        image_id, question, answer = self.data[idx].strip().split(',')
        if not self.use_image_embedding: # If not use embedding, load the image and apply transform
            img = Image.open(f"{self.data_dir}/reformatted/xmlab{int(image_id)}.jpg")
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        else: # if use embedding, directly load the embedding vector for VGG/ResNet
            if self.image_features == None:
                self.image_features = pickle.load(open(os.path.join(self.data_dir, f'{self.mode}_image_embeddings_new_{self.image_model_type}.pkl'), 'rb'))
            img  = self.image_features[image_id]
        
        # convert question words to indexes
        question = [self.word2idx[w] if w in self.word2idx else self.word2idx['<unk>'] for w in question.split()]
        question = pad_sequences(question, self.max_length)

        # convert answer words to indexes
        answer   = self.label2idx[answer if answer in self.label2idx else '<unk>']

        return img, question, answer
