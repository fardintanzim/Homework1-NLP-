
#1.2 Training The Models

from collections import Counter
import math
from itertools import islice

# Function to read corpus from file
def read_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().split()

# Function to train unigram maximum likelihood model
def train_unigram_model(train_corpus):
    unigram_counts = Counter(train_corpus)
    total_tokens = sum(unigram_counts.values())
    unigram_probs = {word: count / total_tokens for word, count in unigram_counts.items()}
    return unigram_probs

# Function to train bigram maximum likelihood model
def train_bigram_model(train_corpus):
    bigram_counts = Counter(zip(train_corpus, train_corpus[1:]))
    bigram_probs = {}
    for bigram, count in bigram_counts.items():
        word1, word2 = bigram
        bigram_probs[bigram] = count / train_corpus.count(word1)
    return bigram_probs

# Function to train bigram model with add-one smoothing
def train_add_one_bigram_model(train_corpus):
    vocabulary_size = len(set(train_corpus))
    bigram_counts = Counter(zip(train_corpus, train_corpus[1:]))
    bigram_probs = {}
    for bigram, count in bigram_counts.items():
        word1, word2 = bigram
        bigram_probs[bigram] = (count + 1) / (train_corpus.count(word1) + vocabulary_size)
    return bigram_probs

# Function to calculate word types and tokens
def calculate_word_types_and_tokens(corpus, exclude_token='<s>'):
    count = Counter(corpus)
    num_types = len(count)
    num_tokens = sum(count.values()) - count[exclude_token]
    return num_types, num_tokens

# Function to calculate unseen percentage
def calculate_unseen_percentage(test_corpus, train_vocab):
    test_count = Counter(test_corpus)
    percentage_unk_types = (test_count["<unk>"] - 1) / (len(test_count) - 1) * 100
    percentage_unk_tokens = test_count["<unk>"] / (sum(test_count.values()) - test_count["<s>"]) * 100
    return percentage_unk_types, percentage_unk_tokens

# Function to calculate unseen bigrams
def calculate_unseen_bigrams(test_corpus, train_vocab):
    test_corpus_unk = ["<unk>" if word not in train_vocab else word for word in test_corpus]
    test_bigrams = list(zip(test_corpus_unk, test_corpus_unk[1:]))
    test_bigram_count = Counter(test_bigrams)

    train_corpus_unk = ["<unk>" if word not in train_vocab else word for word in train_corpus]
    train_bigrams = list(zip(train_corpus_unk, train_corpus_unk[1:]))
    train_bigram_count = Counter(train_bigrams)

    num_unseen_bigram_types = len([bigram for bigram in test_bigram_count if train_bigram_count[bigram] == 0])
    num_unseen_bigram_tokens = sum(test_bigram_count[bigram] for bigram in test_bigram_count if train_bigram_count[bigram] == 0)

    percentage_unseen_bigram_types = (num_unseen_bigram_types / len(test_bigram_count)) * 100
    percentage_unseen_bigram_tokens = (num_unseen_bigram_tokens / len(test_bigrams)) * 100

    return percentage_unseen_bigram_types, percentage_unseen_bigram_tokens

# Function to calculate unigram log probability
def calculate_unigram_log_prob(sentence, train_count):
    return sum(math.log2(train_count[word] / sum(train_count.values())) for word in sentence)

# Function to calculate bigram log probability
def calculate_bigram_log_prob(sentence, bigram_train_count, train_count):
    bigram_input_split = list(zip(sentence, sentence[1:]))
    bigram_log_prob = sum(
        math.log2(bigram_train_count.get(bigram, 0) / train_count.get(bigram[0], 0))
        if bigram_train_count.get(bigram, 0) != 0
        else float("-inf")
        for bigram in bigram_input_split
    )
    return bigram_log_prob if bigram_log_prob != float("-inf") else "undefined"

# Function to calculate add-one bigram log probability
def calculate_add_one_bigram_log_prob(sentence, bigram_train_count, train_count, vocab_size):
    addone_bigram_log_prob = 0
    addone_bigram_input_split = zip(sentence, sentence[1:])
    for bigram in addone_bigram_input_split:
        addone_bigram_log_prob += math.log2((bigram_train_count[bigram] + 1) / (train_count[bigram[0]] + vocab_size))
    return addone_big
