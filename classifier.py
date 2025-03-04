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
import argparse
nltk.download('punkt_tab')

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
    

def split_data(data):
    random.shuffle(data)
    split_index = int(len(data) * 0.9)  # 90% for training, 10% for development
    return data[:split_index], data[split_index:]

def read_author_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.readlines()

def read_test_file(filename):
    text_data = []
    author_labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line:  # Ensure line contains both text and author
                text, author = line.rsplit('|', 1)
                text_data.append(text.strip())
                author_labels.append(author.strip())
    return text_data, author_labels

def save_model(authorlist, models):
    for author in authorlist:
        author_name = os.path.splitext(os.path.basename(author))[0][:-5]
        # Save trained model
        model_file = f"trained_{author_name}.pkl"
        with open(model_file, 'wb') as f:
            lm = models[author_name]
            pickle.dump(lm, f)
        #print(f"Model for {author_name} saved as {model_file}.")

def numeric_labels(labels):
    # Map author labels to human-readable names
    authors = ['austen', 'dickens', 'tolstoy', 'wilde']
    num_labels = len(authors)
    id2label = {i: author for i, author in enumerate(authors)}
    label2id = {author: i for i, author in enumerate(authors)}
    
    # Convert string labels to numeric labels
    numeric_labels = [label2id[label] for label in labels]
    return numeric_labels, num_labels, id2label, label2id

def compute_metrics(eval_pred):
    # Load accuracy metric
    accuracy = evaluate.load('accuracy')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
    
def dev_mode_model(texts,labels,tokenizer,model_name):
    # Convert string labels to numeric labels
    numeric_label, num_labels, id2label, label2id = numeric_labels(labels)
    
    # Create a DataFrame and split into train/test
    df = pd.DataFrame({'text': texts, 'label': numeric_label})
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # Create Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    # Tokenize the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Now set the format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Create the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    return model, train_dataset, test_dataset


def test_mode_model(texts,labels,tokenizer,model_name):
    # Convert string labels to numeric labels
    numeric_label, num_labels, id2label, label2id = numeric_labels(labels)
    
    # Create a DataFrame and split into train/test
    df = pd.DataFrame({'text': texts, 'label': numeric_label})
    #train_df = df

    train_df, test_df = train_test_split(df, test_size=0.00001, random_state=42)
    
    # Create Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    # Tokenize the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Now set the format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    


    # Create the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    return model, train_dataset, test_dataset
##################################################################################################


def argument_parser():
    parser = argparse.ArgumentParser(description="NGram model")
    parser.add_argument('dataset', 
                        type=str, 
                        help="Dataset name (e.g., authorlist)")
    parser.add_argument('-approach', 
                        type=str, 
                        choices=['generative', 'discriminative'], 
                        required=True,
                        help="Approach for classification (generative or discriminative)"
                        )
    parser.add_argument('-test', 
                        type=str, 
                        help="Optional test file for testing sentences")
    args = parser.parse_args()

    authorlist = [line.strip() for line in  open(args.dataset+".txt", "r").readlines()]
    approach = args.approach
    test = args.test
    return authorlist, approach, test


def main():
    authorlist, approach, test = argument_parser()

    if approach == "generative":
        # Load the authors from the provided file
        authorlist = authorlist
        print(authorlist)        
        models = {}
        dev_data = {}
        authors_data = {}
    
        # Read data for each author
        for author_file in authorlist:
            data = read_author_file(author_file)
            authors_data[author_file] = data
        

        if '-test' in sys.argv:
            print("Running in test mode...")
    
            # Use all data for training each author's language model (no train-dev split)
            for author in authorlist:
                author_name = os.path.splitext(os.path.basename(author))[0][:-5]
                data = authors_data[author]
                # Using n=3 everygrams, model_type among: ["Laplace", "Lidstone" , "StupidBackoff",   "WittenBellInterpolated"]
                lm = NGramLanguageModel(n=3, model_type='Laplace')  
   
                print(f"Training {lm.model_type} LM for author: {author_name} on full dataset..."+"\n")
                lm.train(data)
                models[author_name] = lm

                
            save_model(authorlist, models)
    
            # Perform classification on the test data
            test_file = sys.argv[sys.argv.index('-test') + 1]
            test_text, test_authors = read_test_file(test_file)
    
            print("\nPredictions on test set:")
            
            total_sentences = 0
            correct_predictions = 0
            
            for sentence, true_author in zip(test_text, test_authors):
                min_perplexity = float('inf')
                predicted_author = None
                total_sentences += 1
             
                # Compare perplexity for each author model
                for author_name, lm in models.items():
                    perplexity = lm.calculate_perplexity([sentence])
                    if perplexity < min_perplexity:
                        min_perplexity = perplexity
                        predicted_author = author_name
            
                # Print the predicted author for the current sentence
                print(f"Sentence {total_sentences}: Predicted Author -> {predicted_author}, True Author -> {true_author}")
            
                # Check if the prediction is correct
                if predicted_author == true_author:
                    correct_predictions += 1
        
            # Calculate and print test set accuracy
            accuracy = (correct_predictions / total_sentences) * 100 if total_sentences > 0 else 0
            print(f"\nTest Set Accuracy: {accuracy:.2f}%")

            # in a new loop extract and display the top 5 features of each trained model
            for author_name, lm in models.items():
                # Extract and print top features
                top_features = lm.extract_top_features(data)
                
                print(f"Top 5 features with probability scores for {author_name}: {top_features}"+"\n")
             
    
        else:
            # Results on development set
            print("Running in development mode..."+"\n")
    
            # Train models for each author
            print("Splitting into training and development...")
            
            for author in authorlist:
                author_name = os.path.splitext(os.path.basename(author))[0][:-5]
                data = authors_data[author]                                     #balanced_authors_data[author_name]
                train_data, dev_data[author_name] = split_data(data)

                # Using n=3 everygrams, model_type among: ["Laplace", "Lidstone" , "StupidBackoff", "WittenBellInterpolated"]
                lm = NGramLanguageModel(n=3, model_type='Laplace')  # Using trigrams, specify model_type here
                
                print(f"Training {lm.model_type} LM for author: {author_name}..."+"\n")
                lm.train(train_data)
                models[author_name] = lm
    
    
            save_model(authorlist, models)
            print("Results on dev set:")
            total_sentences = 0
            correct_predictions = 0
    
            # Check for each sentence in the development data if the predicted author is correct
            for author in authorlist:
                author_name = os.path.splitext(os.path.basename(author))[0][:-5]
                dev_sentences = dev_data[author_name]
    
                for sentence in dev_sentences:
                    total_sentences += 1
                    min_perplexity = float('inf')
                    predicted_author = None
                    
                    # Find the predicted author (the one with the lowest perplexity)
                    for model_author_name, lm in models.items():
                        perplexity = lm.calculate_perplexity([sentence])
                        if perplexity < min_perplexity:
                            min_perplexity = perplexity
                            predicted_author = model_author_name
                    
                    # Check if the predicted author matches the actual author
                    if predicted_author == author_name:
                        correct_predictions += 1
    
                # Calculate and display the accuracy
                accuracy = (correct_predictions / total_sentences) * 100 if total_sentences > 0 else 0
                print(f"Accuracy {author_name} : {accuracy:.2f}%")

            # in a new loop show the top 5 features of each trained model
            for author_name, lm in models.items():
                # Extract and print top features
                top_features = lm.extract_top_features(data)
                
                print("\n"+f"Top 5 features with probability scores for {author_name}: {top_features}"+"\n")

    elif approach == "discriminative":
        pass

    
if __name__ == "__main__":
    '''
        python3 classifier.py authorlist -approach generative
        python3 classifier.py authorlist -approach generative -test test_sents.txt
    '''
    main()


