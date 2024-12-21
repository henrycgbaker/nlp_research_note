import torch
import torch.nn as nn
import numpy as np

class Collator:
    def __init__(self, vocab_idx, max_seq_length):
        self.vocab_idx = vocab_idx
        self.max_seq_length = max_seq_length

    def __call__(self, data):
        text_list, label_list = [], []
        for _text, _label in data:
            processed_text = torch.tensor(
                [self.vocab_idx[token] for token in _text][:self.max_seq_length],
                dtype=torch.int64
            )
            text_list.append(processed_text)
            label_list.append(_label)
        label_list = torch.tensor(label_list)

        padded_text_list = nn.utils.rnn.pad_sequence(
            text_list,
            batch_first=True,
            padding_value=0
        )
        return padded_text_list, label_list

def embedding_mapping_fasttext(vocabulary, pre_trained_embeddings):
    vocab_size = len(vocabulary)
    embedding_dim = pre_trained_embeddings.get_dimension()
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for idx, word in enumerate(vocabulary):
        embedding_matrix[idx] = pre_trained_embeddings.get_word_vector(word)
    return embedding_matrix