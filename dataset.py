from PIL import Image
from torch.utils.data import Dataset
from utils import pad_sequences
import pydicom as dicom
import cv2
import collections
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
import skimage
from transformers import AutoTokenizer
class VQADataset(Dataset):
  
    def __init__(self, data_dir, transform = None, mode = 'train', use_image_embedding = False, image_model_type = 'torchxrayvision', top_k = 1000, max_length = 14, ignore_unknowns = True, use_softscore = True, use_bert = False, , use_clip = False):
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
        if image_model_type == "torchxrayvision":
            self.transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224, engine="cv2")])
        elif image_model_type == "cass":
            DATASET_IMAGE_MEAN = (0.485, 0.456, 0.406)
            DATASET_IMAGE_STD = (0.229, 0.224, 0.225)
            if mode == "train":
                self.transform = transforms.Compose([transforms.Resize((384,384)),
                    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2),transforms.RandomPerspective(distortion_scale=0.2),], p=0.3),
                    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2),transforms.RandomAffine(degrees=10),], p=0.3),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(DATASET_IMAGE_MEAN, DATASET_IMAGE_STD), ])
            else:
                self.transform = transforms.Compose([transforms.Resize((384,384)),
                    transforms.ToTensor(),
                    transforms.Normalize(DATASET_IMAGE_MEAN, DATASET_IMAGE_STD), ])
        self.mode                  = mode
        self.use_image_embedding   = use_image_embedding
        self.image_model_type      = image_model_type
        
        self.labelfreq             = pickle.load(open(os.path.join(data_dir, f'answers_freqs.pkl'), 'rb'))
        self.label2idx             = {x[0]: i+1 for i, x in enumerate(collections.Counter(self.labelfreq).most_common(n = top_k - 1))}
        self.label2idx["<unk>"]    = 0

        if use_bert:
            self.tokenizer         = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            self.word2idx          = pickle.load(open(os.path.join(data_dir, 'questions_vocab.pkl'), 'rb'))["word2idx"]
        self.max_length            = max_length
        self.ignore_unknowns       = ignore_unknowns
        self.use_softscore         = use_softscore
        
        self.data_file             = f'{mode}_data.txt'

        self.use_clip = use_clip

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
        #print(image_id)
        file_type = image_id.split(".")[-1]
        if file_type == "dcm": # These need to be uncommented for the first time this is being run
            ds = dicom.dcmread(image_id)
            pixel_array_np = ds.pixel_array
            pixel_array_np = 255*pixel_array_np/pixel_array_np.max()
            rgb_image = cv2.cvtColor(np.uint8(pixel_array_np), cv2.COLOR_GRAY2RGB)
            image_id = image_id.replace(".dcm", ".png")
            cv2.imwrite(image_id, rgb_image)
        if not self.use_image_embedding: # If not use embedding, load the image and apply transform

            # Prepare the image:
            if image_model_type == "torchxrayvision":
                img = skimage.io.imread(image_id)
                img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
                #print(img.shape)
                if len(img.shape) == 2:
                    img = img[None, ...]
                elif img.shape[2] == 3:
                    img = img.mean(2)[None, ...] # Make single color channel
            else:
                img = Image.open(image_id)
                img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        else: # if use embedding, directly load the embedding vector for VGG/ResNet
            if self.image_features == None:
                self.image_features = pickle.load(open(os.path.join(self.data_dir+'/clip/', f'{self.mode}_image_embeddings_final.pkl'), 'rb'))
            img  = self.image_features[f"xmlab{int(image_id)}/source.jpg"]
        
        # convert question words to indexes
        if self.use_bert:
            encoded = self.tokenizer.encode_plus(question, max_length=self.max_length, truncation=True, padding='max_length')
            question = {key:torch.LongTensor(value) for key, value in encoded.items()}
        elif self.use_clip:
            question_features = pickle.load(open(os.path.join(self.data_dir+'/clip/', f'{self.mode}_question_embeddings_final.pkl'), 'rb'))
            question  = question_features[question]
        else:
            question = [self.word2idx[w] if w in self.word2idx else self.word2idx['<unk>'] for w in question.split()]
            question = pad_sequences(question, self.max_length)
        # convert answer words to indexes
        answer   = self.label2idx[answer if answer in self.label2idx else '<unk>']

        if self.use_bert:
            return img, question['input_ids'], question['token_type_ids'], question['attention_mask'], answer
        else:
            return img, question, answer


