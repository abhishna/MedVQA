import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import pickle

class ImageEncoder(nn.Module):
    """
    class to encode the image chanel - using either VGG16 or ResNet152 based on user input
    can load pre-saved embeddings from file or perform an inference step with the CNN model
    """

    def __init__(self, output_size = 1024, image_channel_type = 'normi', use_embedding = True, trainable = False,
                 dropout_prob = 0.5, use_dropout = True, image_model_type = 'vgg16'):
        super(ImageEncoder, self).__init__()

        self.image_channel_type = image_channel_type
        self.use_embedding      = use_embedding
        self.image_model_type   = image_model_type
        
        if self.image_model_type == 'resnet152':
            self.model          = models.resnet152(weights = models.ResNet152_Weights.IMAGENET1K_V2)
            self.model          = nn.Sequential(*(list(self.model.children())[:-1]))
        else:
            self.model          = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
            self.model.classifier   = nn.Sequential(*list(self.model.classifier)[:-1])
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.fc    = nn.Sequential()
        if self.image_model_type == 'resnet152':
            self.fc.append(nn.Linear(2048, output_size))
        else:
            self.fc.append(nn.Linear(4096, output_size))
        if use_dropout:
            self.fc.append(nn.Dropout(dropout_prob))
        self.fc.append(nn.Tanh())
    
    def forward(self, images):
        if not self.use_embedding: # Load the image embedding directly
            images      = self.model(images)

        if self.image_model_type == 'resnet152':
            images      = images.flatten(start_dim = 1)

        if self.image_channel_type == 'normi':
            images      = F.normalize(images, p = 2, dim = 1)
        image_embedding = self.fc(images)
        
        return image_embedding

class QuestionEncoder(nn.Module):
    """
    class to encode the text channel, supports GloVe embeddings, vanilla LSTM and bi-directional LSTM based embeddings
    """
    def __init__(self, vocab_size = 10000, word_embedding_size = 300, hidden_size = 512, output_size = 1024,
                 num_layers = 2, dropout_prob = 0.5, use_dropout = True, bi_directional = False, max_seq_len = 14, use_glove = False, use_lstm = False, embedding_file_path = None):
        super(QuestionEncoder, self).__init__()
        
        self.use_glove = use_glove
        self.use_lstm = use_lstm
        self.bi_directional = bi_directional
        self.max_seq_len  = max_seq_len
        self.hidden_size = hidden_size

        if use_glove:
            self.weights_matrix = pickle.load(open(embedding_file_path, 'rb'))
            num_embeddings, embedding_dim = self.weights_matrix.shape
            self.embedding_dim = embedding_dim
            self.word_embeddings = nn.Embedding(num_embeddings, embedding_dim)
            self.word_embeddings.load_state_dict({'weight': torch.from_numpy(self.weights_matrix)})
            self.word_embeddings.weight.requires_grad = False
            

        else:
            self.word_embeddings = nn.Sequential()
            self.word_embeddings.append(nn.Embedding(vocab_size, word_embedding_size, padding_idx = 0))
            if use_dropout:
                self.word_embeddings.append(nn.Dropout(dropout_prob))
        

        if not use_glove:
            self.word_embeddings.append(nn.Tanh())

        self.lstm            = nn.LSTM(input_size = word_embedding_size, hidden_size = hidden_size,
                                       num_layers = num_layers, bidirectional=bi_directional, batch_first=bi_directional )

        self.fc              = nn.Sequential()
        
        
        if bi_directional:
            self.fc.append(nn.Linear(2 * max_seq_len * hidden_size, output_size))
        else:
            self.fc.append(nn.Linear(2 * num_layers * hidden_size, output_size))

        if use_dropout:
            self.fc.append(nn.Dropout(dropout_prob))
        self.fc.append(nn.Tanh())
        
    def forward(self, questions):
        
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
        else:
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
                 image_model_type = 'vgg16', dropout_prob = 0.5, train_cnn = False, use_dropout = True, attention_mechanism = 'element_wise_product', bi_directional=False, max_seq_len = 14, use_glove = False, use_lstm = True, embedding_file_path = None):
        super(VQABaseline, self).__init__()
        
        self.word_embedding_size = word_embedding_size
        
        self.image_encoder       = ImageEncoder(output_size            = embedding_size,
                                                image_channel_type     = image_channel_type,
                                                use_embedding          = use_image_embedding,
                                                trainable              = train_cnn,
                                                dropout_prob           = dropout_prob,
                                                use_dropout            = use_dropout,
                                                image_model_type       = image_model_type)
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
                                                   use_lstm            = use_lstm,
                                                   embedding_file_path = embedding_file_path)
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