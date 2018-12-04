from collections import defaultdict
from math import log
import helper
import numpy as np

# Return a dictionary of words and their correspoding frequency of a training example
def get_freq_of_tokens(words):
    tokens = {}
    for token in words:
        if token not in tokens:
            tokens[token] = 1
        else:
            tokens[token] += 1
    return tokens

# Represent each training example as a vector of the same length.
# Will use the TF-IDF of each word in the vocabulary as feature.
def analyze(training_data):
    nb_of_class0, nb_of_words_class0 = 0, 0
    nb_of_class1, nb_of_words_class1 = 0, 0
    word_count_class0 = defaultdict(int)
    word_count_class1 = defaultdict(int)
    word_count = defaultdict(int)
    occurence = defaultdict(int)
    document_length = []
    
    for idx in range(len(training_data)):
        nb_of_words = 0
        if training_data[idx][1] == "class0":
            nb_of_class0 += 1
            for word, count in training_data[idx][0].items():
                nb_of_words             += count
                word_count_class0[word] += count
                word_count[word]        += count
                occurence[word]         += 1
            nb_of_words_class0 += nb_of_words
        else:
            nb_of_class1 += 1
            for word, count in training_data[idx][0].items():
                nb_of_words             += count
                word_count_class1[word] += count
                word_count[word]        += count
                occurence[word]         += 1
            nb_of_words_class1 += nb_of_words
        document_length.append(nb_of_words)

    vocabulary     = list(word_count.keys())
    vocabulary_idx = {vocabulary[idx]:idx for idx in range(len(vocabulary))}

    # Calculate TF-IDF of each word in the vocabulary for each training example
    x_train, y_train = [], []
    for idx in range(len(training_data)):
        data = [0 for _ in range(len(vocabulary))]
        for word, count in training_data[idx][0].items():
            data[vocabulary_idx[word]] = (count / document_length[idx]) * log((nb_of_class0 + nb_of_class1) / occurence[word], 10)
        x_train.append(data)
        y_train.append(training_data[idx][1])
        
    return x_train, y_train, vocabulary

def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    with open(test_data, 'r') as file:
        test_documents = [line.strip().split(' ') for line in file]
    
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance = helper.strategy()
    parameters = {'gamma':'auto', 'C':1, 'kernel':'linear', 'degree':3, 'coef0':0.0}
    
    ##..................................#
    #
    #
    #
    ## Your implementation goes here....#
    #
    #
    #
    ##..................................#
    training_data = []
    for index in range(len(strategy_instance.class0)):
        training_data.append((get_freq_of_tokens(strategy_instance.class0[index]), "class0"))
    for index in range(len(strategy_instance.class1)):
        training_data.append((get_freq_of_tokens(strategy_instance.class1[index]), "class1"))
    x_train, y_train, vocabulary = analyze(training_data)

    clf = strategy_instance.train_svm(parameters, np.array(x_train), np.array(y_train))
    word_coef = clf.coef_[0].tolist()
    # Map the weights of the trained SVM classifier with their corresponding word in the vocabulary.
    word_coef = {vocabulary[idx]:word_coef[idx] for idx in range(len(vocabulary))}
    
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    modified_test_documents = []
    for document in test_documents:
        # Create a list of words, sorted based on their importance obtained during training.
        # Words that are not seen in the training data will be padded with weights of zero.
        word_coef_test = []
        for word in set(document):
            if word in vocabulary:
                word_coef_test.append((word_coef[word], word))
            else:
                word_coef_test.append((0, word))
        word_coef_test.sort(reverse=True)
        # Create a list of 20 words to delete from the test document
        to_delete = [word_coef_test[idx][1] for idx in range(20)]
    
        modified_document = []
        for word in document:
            if not word in to_delete:
                modified_document.append(word)
        modified_test_documents.append(modified_document)

    with open('modified_data.txt', 'w') as file:
        for document in modified_test_documents:
            for word in document:
                file.write(word + ' ')
            file.write('\n')
        
    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.
