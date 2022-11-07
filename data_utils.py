import fasttext
import math
import config


def load_word2vec_model(model_path: str) -> dict:   
    model_dict = {}

    with open(model_path, 'r', encoding='utf-8') as f:
        number_of_word = int(f.readline())
        size_of_vector = int(f.readline())

        for _ in range(number_of_word):
            line = f.readline()
            splited_line = line[:-2].split(" ")
            word_vector = []

            for j in range(len(splited_line)):
                if j == 0:
                    word = splited_line[j]
                elif j > 1:
                    word_vector.append(float(splited_line[j]))

            assert size_of_vector == len(word_vector), "Length of word vector {} is not equal to {}".format(word, size_of_vector)
            model_dict[word] = word_vector
        
    return model_dict


def load_ft_model(ft_model_path: str):
    return fasttext.load_model(ft_model_path)


def get_word_vector(word: str, model):
    if config.EMBEDDING_MODEL == "w2v":
        return model[word] if word in model else []
    else:
        return model.get_word_vector(word)


def load_train_data(train_data_path: str) -> list:
    word_pair_list = []
    with open(train_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            splited = line[:-1].split(" ")
            word_pair_list.append([splited[0], splited[1]])

    return word_pair_list


def load_test_data(test_data_path: str, task: int) -> list:
    word_pair_label_list = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            splited = line[:-1].split("\t")
            if task == 1:
                word_pair_label_list.append([splited[0], splited[1]]) # word1, word2
            else:
                word_pair_label_list.append([splited[0], splited[1], splited[2]]) # word1, word2, label
            
    return word_pair_label_list[1:]


def export_task1_result(task1_result_dict, oov_count):
    cp = math.floor((len(task1_result_dict) - oov_count)/2 + oov_count) # điểm ở giữa
    
    with open(config.TASK1_RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write("* Độ tương tự thấp: \n")
        f.write("{} - {} : {:.5f}. By sklearn is: {:.5f} \n".format(task1_result_dict[oov_count][0], task1_result_dict[oov_count][1], task1_result_dict[oov_count][2], task1_result_dict[oov_count][3]))
        f.write("{} - {} : {:.5f}. By sklearn is: {:.5f} \n".format(task1_result_dict[oov_count+10][0], task1_result_dict[oov_count+10][1], task1_result_dict[oov_count+10][2], task1_result_dict[oov_count+10][3]))
        f.write("* Độ tương tự trung bình: \n")
        f.write("{} - {} : {:.5f}. By sklearn is: {:.5f} \n".format(task1_result_dict[cp][0], task1_result_dict[cp][1], task1_result_dict[cp][2], task1_result_dict[0][3]))
        f.write("{} - {} : {:.5f}. By sklearn is: {:.5f} \n".format(task1_result_dict[cp+10][0], task1_result_dict[cp+10][1], task1_result_dict[cp+10][2], task1_result_dict[cp+10][3]))
        f.write("* Độ tương tự cao: \n")
        f.write("{} - {} : {:.5f}. By sklearn is: {:.5f} \n".format(task1_result_dict[-1][0], task1_result_dict[-1][1], task1_result_dict[-1][2], task1_result_dict[-1][3]))
        f.write("{} - {} : {:.5f}. By sklearn is: {:.5f} \n".format(task1_result_dict[-10][0], task1_result_dict[-10][1], task1_result_dict[-10][2], task1_result_dict[-10][3]))


def export_task2_result(tg_word1, tg_word2 ,l1, l2, k):
    with open(config.TASK2_RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write("The most {} nearest words with '{}' are: \n".format(k, tg_word1))
        f.write(str(l1) + 2*"\n")
        f.write("The most {} nearest words with '{}' are: \n".format(k, tg_word2))
        f.write(str(l2) + "\n")


# Performance of task3
def export_task3_result(model, test_data_path, p, r, f1):
    with open(config.TASK3_RESULT_PATH, 'a', encoding='utf-8') as f:
        f.write("---------- Performance of {} model in {} ----------".format(str(model).split("(")[0], test_data_path.split("/")[-1]) + "\n")
        f.write("Precision: " + str(p) + "\n")
        f.write("Recall: " + str(r) + "\n")
        f.write("F1: " + str(f1) + 2*"\n")


def export_wrong_prediction(embbed_model, x, y_test, y_pred):
    wrong_pred_list = []
    wrong_pred_count = 0

    assert config.EMBEDDING_MODEL == "w2v", "export_wrong_prediction() in fasttext model is not supported"
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            v1 = x[i][:150]
            v2 = x[i][150:]
            w1, w2 = "", ""
            
            for key, value in embbed_model.items():
                if v1 == value:
                    w1 = key
                if v2 == value:
                    w2 = key

            wrong_pred_count += 1
            wrong_pred_list.append((w1, w2, y_test[i], y_pred[i]))

    with open(config.TASK3_WRONG_PREDICTION_PATH, "w", encoding='utf-8') as f:
        f.write("Number of wrong prediction: " + str(wrong_pred_count) + "\n")
        f.write("Word1 Word2 Label Pred\n")
        for _ in wrong_pred_list:
            f.write("{} {} {} {}\n".format(_[0], _[1], _[2], _[3]))
            