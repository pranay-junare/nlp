import sys
import os
import random
import nltk
from nltk.util import ngrams
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE, WittenBellInterpolated, Lidstone, StupidBackoff
from nltk.lm.vocabulary import Vocabulary
from nltk.lm.models import Laplace, KneserNeyInterpolated

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import transformers

from datasets import Dataset
from collections import Counter
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import evaluate

import wandb
import logging


class NGramLanguageModel:
    def __init__(self, n, model_type='Laplace'):
        self.n = n
        self.model_type = model_type
        
        # Initialize the model based on the type specified
        if model_type == 'MLE':
            self.model = MLE(n,vocabulary = Vocabulary(unk_cutoff=1, unk_label='<UNK>'))
        elif model_type == 'Laplace':
            self.model = Laplace(n, vocabulary = Vocabulary(unk_cutoff=1, unk_label='<UNK>'))
        elif model_type == "Lidstone":
            self.model = Lidstone(gamma=0.05,order=n, vocabulary = Vocabulary(unk_cutoff=1, unk_label='<UNK>'))
        elif model_type == 'WittenBellInterpolated':
            self.model = WittenBellInterpolated(n, vocabulary = Vocabulary(unk_cutoff=1, unk_label='<UNK>'))
        elif model_type == 'StupidBackoff':
            self.model = StupidBackoff(alpha=0.5, order=n, vocabulary = Vocabulary(unk_cutoff=1, unk_label='<UNK>'))  # Note: alpha can be added if needed
        else:
            raise ValueError(f"Unknown model type: {model_type}")
         
    
    def preprocess_data(self, text, n):
        """Tokenize sentences, apply padding, and prepare n-grams."""
        # Tokenize the sentences
        tokenized_sentences = [list(word_tokenize(sentence.lower())) for sentence in text]

        # Create a vocabulary of known words
        known_words = set([word for sentence in tokenized_sentences for word in sentence])
        
        # Handle OOV by replacing words not in the known words set with <UNK>
        processed_sentences = [[word if word in known_words else self.unk_token for word in sentence] 
                               for sentence in tokenized_sentences]
        
        # Padded n-grams , padded_evergram_pipeline returns all possible ngrams generated (unigram+bigram+trigram) for n=3
        train_data, vocab = padded_everygram_pipeline(n, processed_sentences)
        
        return train_data, vocab
        
        
    def train(self, data):
        # Tokenize the sentences and prepare padded n-grams and vocabulary
        train_data, vocab = self.preprocess_data(data, self.n)
        
        # Train the model with padded n-grams
        self.model.fit(train_data, vocab)


    def calculate_perplexity(self, test_data):
        total_log_prob = 0
        total_count = 0

        for sentence in test_data:
            tokens = list(word_tokenize(sentence.lower()))
            test_ngrams = list(nltk.ngrams(pad_both_ends(tokens, n=self.n), self.n))
            
            # Calculate perplexity for each n-gram in the test data
            sentence_prob = self.model.perplexity(test_ngrams)
            total_log_prob += sentence_prob
            total_count += len(test_ngrams)

        # Return perplexity
        perplexity = total_log_prob / total_count if total_count > 0 else float('inf')
        return perplexity

    def generate_text(self, prompt, max_words=50):
        """Generate text given a prompt using the trained language model."""
        # Tokenize the prompt
        prompt_tokens = list(word_tokenize(prompt.lower()))
        
        generated_tokens = prompt_tokens.copy()
    
        # Generate text until we reach the max words limit or encounter an end token
        for _ in range(max_words):
            # Prepare the context from the last n-1 tokens (for an n-gram model)
            ngram_context = generated_tokens[-(self.n-1):]  # n-1 context for n-grams
            next_token = self.model.generate(num_words=max_words, text_seed=ngram_context)
    
            # If the generated token is an end token, stop generating
            if next_token[0] == '</s>':
                break
            
            # Append the generated token to the output
            generated_tokens.append(next_token[0])
    
        # Convert tokens back to string and return
        generated_text = ' '.join(generated_tokens).replace('</s>', '').strip()
        return generated_text

    def extract_top_features(self, text_data, top_n=5):
        """Extract the top N bi-grams from the provided text data."""

        # Create a list of all bi-grams in the text data
        tokenized_sentences = [list(word_tokenize(sentence.lower())) for sentence in text_data]
        all_tokens = [token for sentence in tokenized_sentences for token in sentence]
        
        # Generate bi-grams
        n_grams = list(ngrams(all_tokens, self.n))
        
        # Calculate scores for each bi-gram based on the model
        n_gram_scores = {ng: self.model.score(ng[-1], ng[:-1]) for ng in n_grams}
        
        # Sort the bi-grams by their scores
        sorted_n_grams = sorted(n_gram_scores.items(), key=lambda x: x[1], reverse=True)
        # print(sorted_n_grams)
        
        # Get the top N n-grams based on the model's probabilities
        top_features = sorted_n_grams[:top_n]
        
        return top_features
    
    
def main():
    authorlist_file = str(sys.argv[1])+".txt"
    approach = sys.argv[3]

    print(authorlist_file, approach)
    if approach == "generative":
        pass
    elif approach == "discriminative":
        pass

    
if __name__ == "__main__":
    '''
        python3 classifier.py authorlist -approach generative
        python3 classifier.py authorlist -approach generative -test test_sents.txt
    '''
    main()


