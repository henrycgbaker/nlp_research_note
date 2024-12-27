# !pip install datasets fasttext evaluate
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import pickle


class BidirectionalGRUTextClassifier(nn.Module):
    def __init__(self, embedding_tensor):
        super().__init__()
        # 1) Embedding layer (frozen)
        self.embedding_layer = nn.Embedding.from_pretrained(
            embedding_tensor,
            freeze=True
        )
        
        # 2) 2-layer GRU with dropout + bidirection
        self.gru = nn.GRU(
            input_size=embedding_tensor.size(1),  # embedding dim
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.5,       # dropout only applied between stacked layers
            bidirectional=True
        )
        
        # 3) Classification layer
        #    hidden_size=64 * 2 directions = 128
        self.classification_layer = nn.Linear(in_features=64 * 2, out_features=1)

    def forward(self, x, lengths):
        # x: [batch_size, seq_len]
        # lengths: [batch_size] (lengths of each sequence)

        # 1) Embedding
        x = self.embedding_layer(x)  
        # x now: [batch_size, seq_len, embedding_dim]

        # 2) Pack padded sequence for RNN
        x = nn.utils.rnn.pack_padded_sequence(
            x, 
            lengths.cpu().numpy(), 
            enforce_sorted=False, 
            batch_first=True
        )

        # 3) Forward pass through GRU
        #    - out is a PackedSequence
        #    - h is [num_layers * num_directions, batch_size, hidden_size]
        out, h = self.gru(x)
        
        # For a 2-layer bidirectional GRU:
        # h.shape = [4, batch_size, 64]
        # The top layer forward hidden is h[-2], backward is h[-1]
        forward_top = h[-2, :, :]    # shape: [batch_size, 64]
        backward_top = h[-1, :, :]   # shape: [batch_size, 64]

        # 4) Concatenate the final forward + backward states
        x = torch.cat([forward_top, backward_top], dim=1)  # [batch_size, 128]

        # 5) Classification
        #    (for binary classification, you typically pair with BCEWithLogitsLoss)
        x = self.classification_layer(x)  # [batch_size, 1]

        return x
    
class CNNTextClassifier(nn.Module):
    # create layers
    def __init__(self, embedding_tensor):
        super().__init__()
        # input layer
        self.embedding_layer = nn.Embedding.from_pretrained(embedding_tensor, freeze=True)
        # hidden layers
        convolution_layer = nn.Conv1d(in_channels=embedding_tensor.size(1),
                                      out_channels=128,
                                      kernel_size=3,
                                      padding="same")
        activation_layer = nn.ReLU()
        pooling_layer = nn.AdaptiveAvgPool1d(1)
        h_layers = [convolution_layer, activation_layer, pooling_layer]
        self.hidden_layers = nn.ModuleList(h_layers)
        # classification layer
        self.classification_layer = nn.Linear(in_features=128, out_features=1)

    # define forward pass
    def forward(self, x):
        x = self.embedding_layer(x).permute(0, 2, 1)

        for layer in self.hidden_layers:
            x = layer(x)

        x = x.squeeze(2)

        x = self.classification_layer(x)
        return x
    
class RNNTextClassifier(nn.Module):
    # create layers
    def __init__(self, embedding_tensor):
        super().__init__()
        # input layer
        self.embedding_layer = nn.Embedding.from_pretrained(embedding_tensor, freeze=True)
        # hidden layer
        self.rnn_layer = nn.RNN(input_size=embedding_tensor.size(1),
                                hidden_size=32,
                                num_layers=1, # increase to stack RNNs
                                batch_first=True)
        # classification layer
        self.classification_layer = nn.Linear(in_features=32, out_features=1)

    # define forward pass
    def forward(self, x, lengths):
        x = self.embedding_layer(x)
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              lengths.cpu().numpy(),
                                              enforce_sorted=False,
                                              batch_first=True)
        o_t, h_t = self.rnn_layer(x) # o_t includes the outputs,
                                     # h_t the hidden state at the last time step
        x = h_t[-1, :, :] # extract from last layer (in case of num_layers > 1)
        x = self.classification_layer(x)
        return x
    
class LSTMTextClassifier(nn.Module):
    # create layers
    def __init__(self, embedding_tensor):
        super().__init__()
        # input layer
        self.embedding_layer = nn.Embedding.from_pretrained(embedding_tensor, freeze=True)
        # hidden layer
        self.lstm_layer = nn.LSTM(input_size=embedding_tensor.size(1),
                                  hidden_size=32,
                                  num_layers=1,
                                  batch_first=True)
        # classification layer
        self.classification_layer = nn.Linear(in_features=32, out_features=1)

    # define forward pass
    def forward(self, x, lengths):
        x = self.embedding_layer(x)
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              lengths.cpu().numpy(),
                                              enforce_sorted=False,
                                              batch_first=True)
        o_t, (h_t, c_t) = self.lstm_layer(x) # c_t the cell state at the last time step
        x = h_t[-1, :, :] # extract from last layer (in case of num_layers > 1)
        x = self.classification_layer(x)
        return x
    
class StackedLSTMTextClassifier(nn.Module):
    def __init__(self, embedding_tensor):
        super().__init__()
        # Embedding (frozen)
        self.embedding_layer = nn.Embedding.from_pretrained(
            embedding_tensor, freeze=True
        )
        # LSTM layers
        self.lstm_layer_1 = nn.LSTM(
            input_size=embedding_tensor.size(1),
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.lstm_layer_2 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        # Classification
        self.classification_layer = nn.Linear(in_features=32, out_features=1)

    def forward(self, x, lengths):
        # 1) Embed
        x = self.embedding_layer(x)

        # 2) Pack sequences
        x = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu().numpy(),
            enforce_sorted=False,
            batch_first=True
        )

        # 3) First LSTM
        o_t_1, (h_t_1, c_t_1) = self.lstm_layer_1(x)

        # 4) Directly feed output from LSTM1 to LSTM2 (still packed)
        o_t_2, (h_t_2, c_t_2) = self.lstm_layer_2(o_t_1)

        # 5) Final hidden state (shape: [1, batch_size, 32])
        x = h_t_2[-1, :, :]

        # 6) Classification
        x = self.classification_layer(x)
        return x

class BidirectionalLSTMTextClassifier(nn.Module):
    # create layers
    def __init__(self, embedding_tensor):
        super().__init__()
        # input layer
        self.embedding_layer = nn.Embedding.from_pretrained(embedding_tensor, freeze=True)
        # hidden layer
        self.bid_lstm_layer = nn.LSTM(input_size=embedding_tensor.size(1),
                                      hidden_size=32,
                                      num_layers=1,
                                      batch_first=True,
                                      bidirectional=True)
        # classification layer
        self.classification_layer = nn.Linear(in_features=32*2, out_features=1)

    # define forward pass
    def forward(self, x, lengths):
        x = self.embedding_layer(x)
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              lengths.cpu().numpy(),
                                              enforce_sorted=False,
                                              batch_first=True)
        o_t, (h_t, c_t) = self.bid_lstm_layer(x)
        x = torch.cat((h_t[-2, :, :],
                       h_t[-1, :, :]), dim=1)
        x = self.classification_layer(x)
        return x