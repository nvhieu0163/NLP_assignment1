from math import sqrt
import numpy as np

from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

import config
from data_utils import *
from classifier_model import Classifier


def calculate_similarity(first_word: str, second_word: str, embbed_model):
    first_word_vector = get_word_vector(first_word, embbed_model)
    second_word_vector = get_word_vector(second_word, embbed_model)

    if first_word_vector and second_word_vector: # check if two vector is not null
        if (len(first_word_vector) != len(second_word_vector)): # check if 2 vector is not equal size
            raise Exception("Error in retrival 2 word vector")
        
        numerator = 0
        first_denominator = 0
        second_denominator = 0

        for i in range(len(first_word_vector)):
            numerator += (first_word_vector[i] * second_word_vector[i])
            first_denominator += pow(first_word_vector[i], 2)
            second_denominator += pow(second_word_vector[i], 2)
        
        return numerator/(sqrt(first_denominator)*sqrt(second_denominator))       
    else:
        return "Unknown"


def find_K_nearest_word(target_word: str, _k: int, embbed_model) -> list:
    if config.EMBEDDING_MODEL == "w2v":
        word_dict = {}
        word_list = []

        for word in embbed_model:
            word_dict[word] = calculate_similarity(target_word, word, embbed_model)

        sorted_dict = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True))
        
        for word in sorted_dict:
            if len(word_list) > _k+1:
                break
            else: 
                word_list.append((word, sorted_dict[word]))

        return word_list[1:]
    else:
        return embbed_model.get_nearest_neighbors(target_word)[:_k] # fasttext pretrain model


def main():
    
    print("!! Set configuration in 'config.py' first")
    if config.EMBEDDING_MODEL == "w2v":
        embbed_model = load_word2vec_model(config.WORD2VEC_MODEL_PATH) #word2vec model
    else: 
        embbed_model = load_ft_model(config.FASTTEXT_MODEL_PATH) #fasttext embedding model


    ###############################  BÀI I  ###############################
    print("=" * 20 + " Bài 1 " + "=" * 20)
    
    task1_data = load_test_data(config.TASK1_TEST_DATA_PATH, task=1)
    oov_count = 0

    for data in task1_data:
        similarity = calculate_similarity(data[0], data[1], embbed_model) 

        if isinstance(similarity, float):
            vec1 = np.array([get_word_vector(data[0], embbed_model)])
            vec2 = np.array([get_word_vector(data[1], embbed_model)])
            similarity_by_sklearn = cosine_similarity(vec1, vec2)[0][0] # using sklearn library

            data.extend([similarity, similarity_by_sklearn]) 
        else:
            oov_count += 1
            data.extend([-1, -1]) # no word vector in pretrain model

    task1_data.sort(key=lambda x: x[2])

    export_task1_result(task1_data, oov_count)
    print("Result in: '{}'".format(config.TASK1_RESULT_PATH), "\n")
    

    ###############################  BÀI II  ###############################
    print("=" * 20 + " Bài 2 " + "=" * 20)

    target_word_1 = input("first target word: ")
    target_word_2 = input("second target word: ")
    k_parameter = int(input("K paramater: "))

    l1 = find_K_nearest_word(target_word_1, k_parameter, embbed_model)
    l2 = find_K_nearest_word(target_word_2, k_parameter, embbed_model)
    export_task2_result(target_word_1, target_word_2 ,l1, l2, k_parameter)
    print("Result in: '{}'".format(config.TASK2_RESULT_PATH), "\n")


    ###############################  BÀI IIII  ###############################
    print("=" * 20 + " Bài 3 " + "=" * 20)
    base_model = MLPClassifier()
    classifier_model = Classifier(embbed_model, base_model)

    # load data
    antonym_data = load_train_data(config.ANTONYM_DATA_PATH)
    synonym_data = load_train_data(config.SYNONYM_DATA_PATH)
    test_data = load_test_data(config.TASK3_TEST_DATA_PATH, task=3)

    # embeddings
    x_train, y_train = classifier_model.generate_train_data(antonym_data, synonym_data)
    x_test, y_test = classifier_model.generate_test_data(test_data)
    
    # shuffle training data
    _x, _y = shuffle(x_train, y_train)

    # training
    classifier_model.train(_x, _y)

    # predict
    y_pred = classifier_model.predict(x_test)

    # evaluate
    p, r, f1 = classifier_model.evalutate(y_pred, y_test)
    _p, _r, _f1 = classifier_model.evaluate_by_sklearn(y_pred, y_test)

    print("---------- Performance of {} model in {} ----------".format(str(base_model).split("(")[0], config.TASK3_TEST_DATA_PATH.split("/")[-1]))
    print("Precision: \t", p, "\t", _p)
    print("Recall: \t", r, "\t", _r)
    print("F1: \t", f1, "\t", _f1)

    export_task3_result(base_model, config.TASK3_TEST_DATA_PATH, p, r, f1)
    export_wrong_prediction(embbed_model, x_test, y_test, y_pred)

if __name__ == "__main__":
    main()