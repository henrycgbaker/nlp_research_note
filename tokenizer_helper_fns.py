# misinfo_tokenizer.py

import os
import pickle
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

nlp = spacy.load("en_core_web_sm", 
                 disable=["tok2vec", "tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])

def custom_tokenizer(text):
    tokenized_text = nlp(text)
    return [tok.text for tok in tokenized_text]

def custom_analyzer(text, trained_tokenizer):
    """
    Uses the custom_tokenizer, then replaces out-of-vocabulary tokens with <unk>.
    """
    tokens = custom_tokenizer(text)
    vocab = trained_tokenizer.vocabulary_
    return [token if token in vocab else "<unk>" for token in tokens]

def get_trained_tokenizer(text_series, rnn_tokenizer_file=None, min_df=3):
    """
    1) Checks if a previously fitted tokenizer exists in tokenizer_file.
    2) If not, create a new CountVectorizer, fit it on 'text_series'.
    3) Save the fitted tokenizer if tokenizer_file is provided.
    4) Return the tokenizer.
    """
    # Ensure the directory exists, but check if it's a file first
    if rnn_tokenizer_file:
        tokenizer_dir = os.path.dirname(rnn_tokenizer_file)
        if os.path.exists(tokenizer_dir):
            if os.path.isfile(tokenizer_dir):
                raise FileExistsError(f"The path '{tokenizer_dir}' exists but is a file, not a directory.")
        else:
            os.makedirs(tokenizer_dir, exist_ok=True)

    # If a tokenizer file path is given and exists, load it
    if rnn_tokenizer_file and os.path.exists(rnn_tokenizer_file):
        print(f"Tokenizer file '{rnn_tokenizer_file}' found. Loading it...")
        with open(rnn_tokenizer_file, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        # Otherwise, create a new one and fit
        print("No pre-fitted tokenizer found or no file specified. Creating a new one...")
        tokenizer = CountVectorizer(
            analyzer="word",
            tokenizer=custom_tokenizer,  # We define custom_tokenizer for splitting
            lowercase=False,
            min_df=min_df
        )
        tokenizer.fit(text_series)
        
        # Save the tokenizer if a path was provided
        if rnn_tokenizer_file:
            print(f"Saving fitted tokenizer to '{rnn_tokenizer_file}'...")
            with open(rnn_tokenizer_file, 'wb') as f:
                pickle.dump(tokenizer, f)

    return tokenizer

def batch_tokenize(text_series, batch_size, analyzer_func):
    """
    Tokenizes a Pandas Series of text in batches to avoid memory issues.
    """
    tokenized_result = []
    total = len(text_series)
    num_batches = (total // batch_size) + (1 if total % batch_size != 0 else 0)
    
    for batch_idx in range(0, total, batch_size):
        
        # Print progress every 200 batches or at the last batch
        if (batch_idx // batch_size + 1) % 200 == 0 or (batch_idx + batch_size >= total):
            print(f'Tokenizing batch {batch_idx // batch_size + 1} of {num_batches}...')
        
        batch_texts = text_series[batch_idx : batch_idx + batch_size]
        for text in batch_texts:
            tokenized_result.append(analyzer_func(text))
    
    return tokenized_result

def vocab_mapping(tokenized_text):
    token_counts = Counter()
    for text in tokenized_text:
        token_counts.update(text)
    special_tokens = ["<pad>", "<unk>"]
    vocab_tokens = special_tokens + [token for token, freq in token_counts.most_common()]
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
    return vocab

def vocab_mapping(tokenized_text):
    token_counts = Counter()
    for text in tokenized_text:
        token_counts.update(text)
    special_tokens = ["<pad>", "<unk>"]
    vocab_tokens = special_tokens + [token for token, freq in token_counts.most_common()]
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
    return vocab