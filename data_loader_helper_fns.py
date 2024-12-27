import torch
import torch.nn as nn
import numpy as np

def collate_fn(data, include_lengths=True):
    text_list, label_list, lengths = [], [], []
    for _text, _label in data:
        # Integer encoding with truncation
        processed_text = torch.tensor([vocab_idx[token] for token in _text][:max_seq_length],
                                      dtype=torch.int64)
        text_list.append(processed_text)
        label_list.append(_label)
        lengths.append(processed_text.size(0))
    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    # Padding
    padded_text_list = nn.utils.rnn.pad_sequence(text_list,
                                                 batch_first=True,
                                                 padding_value=0)
    if include_lengths:
        return padded_text_list, label_list, lengths
    else:
        return padded_text_list, label_list

def embedding_mapping_fasttext(vocabulary, pre_trained_embeddings):
    vocab_size = len(vocabulary)
    embedding_dim = pre_trained_embeddings.get_dimension()
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for idx, word in enumerate(vocabulary):
        embedding_matrix[idx] = pre_trained_embeddings.get_word_vector(word)
    return embedding_matrix