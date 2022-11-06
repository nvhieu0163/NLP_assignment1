from data_utils import get_word_vector
from sklearn.metrics import precision_score, recall_score, f1_score

class Classifier():
    def __init__(self, embeddings_model, base_model):
        self.embeddings_model = embeddings_model
        self.base_model = base_model


    def generate_train_data(self, antonym_data, synonym_data):
        x_train = []
        y_train = []

        for word_pair in synonym_data:
            feature = self.get_feature(word_pair[0], word_pair[1])
            if not feature:
                continue
            else:
                y_train.extend([1])
                x_train.append(feature)

        for word_pair in antonym_data:
            feature = self.get_feature(word_pair[0], word_pair[1])
            if not feature:
                continue
            else:
                y_train.extend([0])
                x_train.append(feature)

        print("The ratio of label 1:0 in train data is {:.2f}:1".format(y_train.count(1)/y_train.count(0)))
        assert len(x_train) ==  len(y_train), "Error in generating train data"
        return x_train, y_train
    

    def generate_test_data(self, test_data):
        x_test = []
        y_test = []

        for word_pair in test_data:
            feature = self.get_feature(word_pair[0], word_pair[1])
            if not feature:
                continue
            else:
                y_test.extend([1] if word_pair[2] == "SYN" else [0])
                x_test.append(feature)

        print("The ratio of label 1:0 in test data is {:.2f}:1".format(y_test.count(1)/y_test.count(0)))
        assert len(x_test) ==  len(y_test), "Error in generating test data"
        return x_test, y_test


    def get_feature(self, first_word, second_word):
        vector_1 = get_word_vector(first_word, self.embeddings_model)
        vector_2 = get_word_vector(second_word, self.embeddings_model)

        if vector_1 and vector_2:
            return vector_1 + vector_2
        else:
            return []


    def train(self, x_train, y_train):
        self.base_model.fit(x_train, y_train)


    def predict(self, x_test):
        return self.base_model.predict(x_test)


    def evalutate(self, y_predict, y_test):
        assert len(y_predict) == len(y_test), "y_predict is not equal size to y_test"
        true_positive, false_positive, false_negative = 0, 0, 0
        p, r, f1 = 0, 0, 0

        for i in range(len(y_predict)):
            if y_predict[i] ==  1 and y_test[i] == 1:
                true_positive += 1
            elif y_predict[i] == 1 and y_test[i] == 0:
                false_positive += 1
            elif y_predict[i] == 0 and y_test[i] == 1:
                false_negative += 1

        p = true_positive / (true_positive + false_positive)
        r = true_positive / (true_positive + false_negative)
        f1 = 2*p*r / (p + r)
        return p, r, f1


    def evaluate_by_sklearn(self, y_predict, y_test):
        p = precision_score(y_test, y_predict)
        r = recall_score(y_test, y_predict)
        f1 = f1_score(y_test, y_predict)

        return p, r, f1