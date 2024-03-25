# Question 5,6,7

from collections import Counter
import math

# Function to calculate unigram log probability with smoothing
def calculate_unigram_log_prob(sentence, train_count, smoothing_term=1e-10):
    total_count = sum(train_count.values())
    return sum(math.log2((train_count[word] + smoothing_term) / (total_count + len(train_count) * smoothing_term)) for word in sentence)

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
    return addone_bigram_log_prob

# Function to calculate corpus perplexity
def calculate_corpus_perplexity(log_probs, num_tokens):
    total_log_prob = sum(log_prob if log_prob != "undefined" else float("-inf") for log_prob in log_probs)
    if total_log_prob == float("-inf"):
        return "undefined"
    return 2 ** (-1 * total_log_prob / num_tokens)

# Sample sentence
sentence = ["<s>", "i", "look", "forward", "to", "hearing", "your", "reply", ".", "</s>"]

# Assuming train_corpus is defined
train_corpus = ["<s>", "i", "look", "forward", "to", "hearing", "your", "reply", ".", "</s>"]

# Unigram model
train_count_unigram = Counter(train_corpus)
unigram_log_prob = calculate_unigram_log_prob(sentence, train_count_unigram)
print("Question 5: Log Probability for the sentence under Unigram model:", unigram_log_prob)

# Bigram model
bigram_train_count = Counter(zip(train_corpus, train_corpus[1:]))
bigram_log_prob = calculate_bigram_log_prob(sentence, bigram_train_count, train_count_unigram)
print("Question 5: Log Probability for the sentence under Bigram model:", bigram_log_prob)

# Add-One Bigram model
vocab_size = len(train_count_unigram)
add_one_bigram_log_prob = calculate_add_one_bigram_log_prob(sentence, bigram_train_count, train_count_unigram, vocab_size)
print("Question 5: Log Probability for the sentence under Add-One Bigram model:", add_one_bigram_log_prob)

# Calculate perplexity for the sample sentence under each model
num_tokens = len(sentence) - 1  # excluding <s>
unigram_perplexity = calculate_corpus_perplexity([unigram_log_prob], num_tokens)
print("Question 6: Perplexity for the sentence under Unigram model:", unigram_perplexity)

bigram_perplexity = calculate_corpus_perplexity([bigram_log_prob], num_tokens)
print("Question 6: Perplexity for the sentence under Bigram model:", bigram_perplexity)

add_one_bigram_perplexity = calculate_corpus_perplexity([add_one_bigram_log_prob], num_tokens)
print("Question 6: Perplexity for the sentence under Add-One Bigram model:", add_one_bigram_perplexity)

# Assuming test_corpus is defined
test_corpus = [["<s>", "this", "is", "a", "test", "sentence", ".", "</s>"],
               ["<s>", "another", "test", "sentence", ".", "</s>"]]

# Calculate perplexity for the test corpus under each model
test_num_tokens = sum(len(sentence) - 1 for sentence in test_corpus)  # excluding <s>
unigram_test_log_probs = [calculate_unigram_log_prob(sentence, train_count_unigram) for sentence in test_corpus]
unigram_test_perplexity = calculate_corpus_perplexity(unigram_test_log_probs, test_num_tokens)
print("Question 7: Perplexity for the test corpus under Unigram model:", unigram_test_perplexity)

bigram_test_log_probs = [calculate_bigram_log_prob(sentence, bigram_train_count, train_count_unigram) for sentence in test_corpus]
bigram_test_perplexity = calculate_corpus_perplexity(bigram_test_log_probs, test_num_tokens)
print("Question 7: Perplexity for the test corpus under Bigram model:", bigram_test_perplexity)

add_one_bigram_test_log_probs = [calculate_add_one_bigram_log_prob(sentence, bigram_train_count, train_count_unigram, vocab_size) for sentence in test_corpus]
add_one_bigram_test_perplexity = calculate_corpus_perplexity(add_one_bigram_test_log_probs, test_num_tokens)
print("Question 7: Perplexity for the test corpus under Add-One Bigram model:", add_one_bigram_test_perplexity)
