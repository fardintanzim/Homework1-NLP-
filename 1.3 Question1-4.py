#Question 1,2,3,4

from collections import Counter

# Function to read corpus from file
def read_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().split()

# Function to preprocess corpus
def preprocess_corpus(corpus):
    preprocessed_corpus = []
    for sentence in corpus:
        preprocessed_sentence = ["<s>"] + sentence.lower().split() + ["</s>"]
        preprocessed_corpus.append(preprocessed_sentence)
    return preprocessed_corpus

# Function to calculate word types and tokens
def calculate_word_types_and_tokens(corpus):
    word_counts = Counter(word for sentence in corpus for word in sentence)
    num_word_types = len(word_counts) + 1  # Adding for </s>
    num_word_tokens = sum(word_counts.values()) - word_counts['<s>']
    return num_word_types, num_word_tokens

# Function to calculate unseen word types and tokens
def calculate_unseen_word_types_and_tokens(test_corpus, train_vocab):
    test_words = [word for sentence in test_corpus for word in sentence]
    unseen_word_types = sum(1 for word in set(test_words) if word not in train_vocab)
    unseen_word_tokens = sum(1 for word in test_words if word not in train_vocab and word != '<s>')
    total_word_tokens = len(test_words) - test_words.count('<s>')
    percentage_unseen_word_types = (unseen_word_types / len(set(test_words))) * 100
    percentage_unseen_word_tokens = (unseen_word_tokens / total_word_tokens) * 100
    return percentage_unseen_word_types, percentage_unseen_word_tokens

## Function to calculate unseen bigrams
def calculate_unseen_bigrams(test_corpus, train_bigrams):
    test_bigrams = []
    for sentence in test_corpus:
        sentence_bigrams = [(sentence[i], sentence[i + 1]) for i in range(len(sentence) - 1)]
        test_bigrams.extend(sentence_bigrams)
    test_bigrams = [tuple(bigram) for bigram in test_bigrams]  # Convert lists to tuples
    unseen_bigram_types = sum(1 for bigram in set(test_bigrams) if bigram not in train_bigrams)
    unseen_bigram_tokens = sum(1 for bigram in test_bigrams if bigram not in train_bigrams)
    total_bigram_tokens = len(test_bigrams)
    percentage_unseen_bigram_types = (unseen_bigram_types / len(set(test_bigrams))) * 100
    percentage_unseen_bigram_tokens = (unseen_bigram_tokens / total_bigram_tokens) * 100
    return percentage_unseen_bigram_types, percentage_unseen_bigram_tokens




train_file_path = r"C:\Spring2024\csci 366 nlp\pythonProject366\train_processed.txt"
test_file_path = r"C:\Spring2024\csci 366 nlp\pythonProject366\test_processed.txt"

# Read training and test corpora
train_corpus = read_corpus(train_file_path)
test_corpus = read_corpus(test_file_path)

# Preprocess corpora
train_corpus = preprocess_corpus(train_corpus)
test_corpus = preprocess_corpus(test_corpus)

# Calculate word types and tokens in the training corpus
num_word_types, num_word_tokens = calculate_word_types_and_tokens(train_corpus)
print("Question 1: Number of word types in the training corpus (including </s> and <unk>):", num_word_types)
print("Question 2: Number of word tokens in the training corpus (excluding <s>):", num_word_tokens)

# Calculate percentage of unseen word types and tokens in the test corpus
train_vocab = set(word for sentence in train_corpus for word in sentence)
percentage_unseen_word_types, percentage_unseen_word_tokens = calculate_unseen_word_types_and_tokens(test_corpus, train_vocab)
print("Question 3:")
print("Percentage of unseen word types: {:.2f}%".format(percentage_unseen_word_types))
print("Percentage of unseen word tokens: {:.2f}%".format(percentage_unseen_word_tokens))

# Calculate percentage of unseen bigram types and tokens in the test corpus
train_bigrams = set((sentence[i], sentence[i + 1]) for sentence in train_corpus for i in range(len(sentence) - 1))
percentage_unseen_bigram_types, percentage_unseen_bigram_tokens = calculate_unseen_bigrams(test_corpus, train_bigrams)
print("Question 4:")
print("Percentage of bigram types in the test corpus that did not occur in training: {:.2f}%".format(percentage_unseen_bigram_types))
print("Percentage of bigram tokens in the test corpus that did not occur in training: {:.2f}%".format(percentage_unseen_bigram_tokens))
