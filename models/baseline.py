import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import pickle
import torchxrayvision as xrv
import timm
from transformers import AutoModel


class ImageEncoder(nn.Module):
    """
    class to encode the image chanel - using either VGG16 or ResNet152 based on user input
    can load pre-saved embeddings from file or perform an inference step with the CNN model
    """

    def __init__(self, output_size = 1024, image_channel_type = 'normi', use_embedding = True, trainable = False,
                 dropout_prob = 0.5, use_dropout = True, image_model_type = 'torchxrayvision', use_clip = False):
        super(ImageEncoder, self).__init__()

        self.image_channel_type = image_channel_type
        self.use_embedding      = use_embedding
        self.image_model_type   = image_model_type

        if self.image_model_type == 'torchxrayvision':
            #print("using pretrained models on xray images")
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.model = nn.Sequential(*(list(self.model.children())[:-2]))
            #self.model = xrv.autoencoders.ResNetAE(weights="101-elastic")
        
        elif self.image_model_type == 'resnet152':
            self.model          = models.resnet152(weights = models.ResNet152_Weights.IMAGENET1K_V2)
            self.model          = nn.Sequential(*(list(self.model.children())[:-1]))
        elif self.image_model_type == 'cass':
            self.model = timm.create_model('vit_base_patch16_384', pretrained=True)
            ckpt = torch.load('./cass_vit.pt')
            self.model.load_state_dict(ckpt.module.state_dict())
        else:
            self.model          = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
            self.model.classifier   = nn.Sequential(*list(self.model.classifier)[:-1])
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.fc    = nn.Sequential()

        if self.image_model_type == 'torchxrayvision':
            #print("making fc for new model")
            self.fc.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.fc.append(nn.Flatten())
            self.fc.append(nn.Linear(1024, output_size))

        elif self.image_model_type == 'resnet152':
            self.fc.append(nn.Linear(2048, output_size))
        elif self.image_model_type == 'vgg16':
            self.fc.append(nn.Linear(4096, output_size))
        elif self.image_model_type == 'cass':
            self.fc.append(nn.Linear(1000, output_size))
        elif use_clip:
            self.fc.append(nn.Linear(512, output_size))
        if use_dropout:
            self.fc.append(nn.Dropout(dropout_prob))
        self.fc.append(nn.Tanh())
    
    def forward(self, images):
        #if not self.use_embedding: # Load the image embedding directly
        
        images      = self.model(images)
        #images = self.model.encode(images)

        if self.image_model_type == 'resnet152':
            images      = images.flatten(start_dim = 1)

        if self.image_channel_type == 'normi': # COMMENT THIS OUT WHEN USING TORCHXRAYVISION
            images      = F.normalize(images, p = 2, dim = 1) 
        image_embedding = self.fc(images)
        
        return image_embedding

class QuestionEncoder(nn.Module):
    """
    class to encode the text channel, supports GloVe embeddings, vanilla LSTM and bi-directional LSTM based embeddings
    """
    def __init__(self, vocab_size = 10000, word_embedding_size = 300, hidden_size = 512, output_size = 1024,
                 num_layers = 2, dropout_prob = 0.5, use_dropout = True, bi_directional = False, max_seq_len = 14, use_glove = False, use_lstm = False, use_bert = False, embedding_file_path = None, use_clip = False):
        super(QuestionEncoder, self).__init__()
        
        self.use_glove = use_glove
        self.use_lstm = use_lstm
        self.bi_directional = bi_directional
        self.max_seq_len  = max_seq_len
        self.hidden_size = hidden_size
        self.use_clip = use_clip
        self.use_bert = use_bert

        if use_glove:
            self.weights_matrix = pickle.load(open(embedding_file_path, 'rb'))
            num_embeddings, embedding_dim = self.weights_matrix.shape
            self.embedding_dim = embedding_dim
            self.word_embeddings = nn.Embedding(num_embeddings, embedding_dim)
            self.word_embeddings.load_state_dict({'weight': torch.from_numpy(self.weights_matrix)})
            self.word_embeddings.weight.requires_grad = False
            
        elif use_bert:
            self.bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",output_hidden_states=True)

        else:
            self.word_embeddings = nn.Sequential()
            self.word_embeddings.append(nn.Embedding(vocab_size, word_embedding_size, padding_idx = 0))
            if use_dropout:
                self.word_embeddings.append(nn.Dropout(dropout_prob))
        

        if not use_glove and not use_bert:
            self.word_embeddings.append(nn.Tanh())

        self.lstm            = nn.LSTM(input_size = word_embedding_size, hidden_size = hidden_size,
                                       num_layers = num_layers, bidirectional=bi_directional, batch_first=bi_directional )

        self.fc              = nn.Sequential()
        
        
        if use_clip:
            self.fc.append(nn.Linear(512, output_size))
        elif bi_directional:
            self.fc.append(nn.Linear(2 * max_seq_len * hidden_size, output_size))
        elif use_bert:
            self.fc.append(nn.Linear(3072, output_size))
        else:
            self.fc.append(nn.Linear(2 * num_layers * hidden_size, output_size))

        if use_dropout:
            self.fc.append(nn.Dropout(dropout_prob))
        self.fc.append(nn.Tanh())
        
    def forward(self, questions):

        if(self.use_clip):
            return self.fc(questions)
        
        if self.use_bert:
            x = self.bert_model(input_ids = questions[0], token_type_ids = questions[1], attention_mask = questions[2])
            hidden_states = x[2]
            last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
            # cast layers to a tuple and concatenate over the last dimension
            cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
            x = torch.mean(cat_hidden_states, dim=1).squeeze()
        else:
            x                  = self.word_embeddings(questions)
        if self.use_lstm:
            if self.bi_directional == False:
                # N * seq_length * 300
                x                  = x.transpose(0, 1)
                # seq_length * N * 300
                _, (hidden, cell)  = self.lstm(x)
                # (1 * N * 1024, 1 * N * 1024)
                x                  = torch.cat((hidden, cell), 2)
                # (1 * N * 2048)
                x                  = x.transpose(0, 1)
                # (N * 1 * 2048)
                x                  = x.reshape(x.size()[0], -1)
                # (N * 2048)
                x                  = nn.Tanh()(x)
            else:
                x, (hidden, cell)  = self.lstm(x)
                x = x.reshape(-1,2*self.max_seq_len*self.hidden_size)
        elif not self.use_bert:
            x = x.reshape(-1,self.max_seq_len*self.embedding_dim)
        question_embedding = self.fc(x)
        # (N * 1024)

        return question_embedding

class VQABaseline(nn.Module):
    """
    class to perform a forward pass through the model
    get embeddings for image and text channel
    combine them using the specified attention mechanism
    pass the joint vector representation via a deep NN classifier to get the output answer
    """
    def __init__(self, vocab_size = 10000, word_embedding_size = 300, embedding_size = 1024, output_size = 1000,
                 lstm_hidden_size = 512, num_lstm_layers = 2, image_channel_type = 'normi', use_image_embedding = True,
                 image_model_type = 'torchxrayvision', dropout_prob = 0.5, train_cnn = False, use_dropout = True, attention_mechanism = 'element_wise_product', bi_directional=False, max_seq_len = 14, use_glove = False, use_lstm = True, use_bert = False, embedding_file_path = None, use_clip = False):
        super(VQABaseline, self).__init__()
        
        self.word_embedding_size = word_embedding_size
        
        self.image_encoder       = ImageEncoder(output_size            = embedding_size,
                                                image_channel_type     = image_channel_type,
                                                use_embedding          = use_image_embedding,
                                                trainable              = train_cnn,
                                                dropout_prob           = dropout_prob,
                                                use_dropout            = use_dropout,
                                                image_model_type       = image_model_type,
                                                use_clip=use_clip)
        self.question_encoder    = QuestionEncoder(vocab_size          = vocab_size,
                                                   word_embedding_size = word_embedding_size,
                                                   hidden_size         = lstm_hidden_size,
                                                   output_size         = embedding_size,
                                                   num_layers          = num_lstm_layers,
                                                   dropout_prob        = dropout_prob,
                                                   use_dropout         = use_dropout,
                                                   bi_directional      = bi_directional,
                                                   max_seq_len         = max_seq_len,
                                                   use_glove           = use_glove,
                                                   use_bert            = use_bert,
                                                   use_lstm            = use_lstm,
                                                   embedding_file_path = embedding_file_path,
                                                   use_clip=use_clip)
        self.attention_mechanism = attention_mechanism
        self.attention_fn = {'element_wise_product': lambda x,y:x*y, 'sum': torch.add, 'concat': lambda x,y:torch.cat((x,y),dim=1)}
        self.embedding_size_post_attention = {'element_wise_product': embedding_size, 'sum': embedding_size, 'concat': 2*embedding_size}
        self.mlp                 = nn.Sequential()
        self.mlp.append(nn.Linear(self.embedding_size_post_attention[self.attention_mechanism], 1000))
        self.mlp.append(nn.Dropout(dropout_prob)) # part of the base line model by default
        self.mlp.append(nn.Tanh())
        self.mlp.append(nn.Linear(1000, output_size))
        
        

    def forward(self, images, questions):
        image_embeddings    = self.image_encoder(images) # Image Embeddings
        question_embeddings = self.question_encoder(questions) # Question Embeddings
        final_embedding     = self.attention_fn[self.attention_mechanism](image_embeddings, question_embeddings) # Attention
        
        output              = self.mlp(final_embedding) # Classifier
        
        return output
