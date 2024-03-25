
# 1.1 Pre-Processing

train_file_path = r"C:\Spring2024\csci 366 nlp\hw1\train-Spring2024.txt"
test_file_path = r"C:\Spring2024\csci 366 nlp\hw1\test.txt"

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def preprocess_sentence(sentence):
    # Add start and end symbols
    sentence = '<s> ' + sentence.strip().lower() + ' </s>'
    return sentence

def replace_singletons_with_unk(corpus):
    word_freq = {}
    for sentence in corpus:
        for word in sentence.split():
            word_freq[word] = word_freq.get(word, 0) + 1

    processed_corpus = []
    for sentence in corpus:
        processed_sentence = []
        for word in sentence.split():
            if word_freq[word] == 1:
                processed_sentence.append('<unk>')
            else:
                processed_sentence.append(word)
        processed_corpus.append(' '.join(processed_sentence))
    return processed_corpus, set(word_freq.keys())

def write_to_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data))

# Step 1: Read files
train_corpus = read_file(train_file_path)
test_corpus = read_file(test_file_path)

# Step 2: Pre-processing
train_corpus_processed = [preprocess_sentence(sentence) for sentence in train_corpus]
test_corpus_processed = [preprocess_sentence(sentence) for sentence in test_corpus]

# Step 3: Replace singletons with <unk> and get unique words
train_corpus_processed, train_unique_words = replace_singletons_with_unk(train_corpus_processed)
test_corpus_processed, test_unique_words = replace_singletons_with_unk(test_corpus_processed)

# Step 4: Write unique words to a new file
unique_words_file_path = "unique_words.txt"
unique_words = train_unique_words.union(test_unique_words)
write_to_file(unique_words_file_path, unique_words)

# Step 5: Write pre-processed data to new files
write_to_file('train_processed.txt', train_corpus_processed)
write_to_file('test_processed.txt', test_corpus_processed)

# Sample output for verification
print("Pre-processed train corpus sample:")
print(train_corpus_processed[:3])
print("\nPre-processed test corpus sample:")
print(test_corpus_processed[:3])
print("\nTotal unique words:", len(unique_words))
