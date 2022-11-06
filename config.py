
#############################################################################
##### Configuration of program #####
#############################################################################

# Choose pretrain model for embedding.
EMBEDDING_MODEL = "w2v"  # "w2v" for word2vec / "ft" for fasttext

## PATH for word2vec / embeddings model
WORD2VEC_MODEL_PATH = "./pretrain_models/W2V_150.txt"
FASTTEXT_MODEL_PATH = "./pretrain_models/cc.vi.150.bin"

## PATH for datasets
ANTONYM_DATA_PATH = "./datasets/Antonym_vietnamese.txt"
SYNONYM_DATA_PATH = "./datasets/Synonym_vietnamese.txt"

TASK1_TEST_DATA_PATH = "./datasets/Visim-400.txt"
TASK3_TEST_DATA_PATH = "./datasets/400_verb_pairs.txt"

## PATH for task output result
TASK1_RESULT_PATH = "./experiment_result/task1.txt"
TASK2_RESULT_PATH = "./experiment_result/task2.txt"
TASK3_RESULT_PATH = "./experiment_result/task3.txt"

TASK3_WRONG_PREDICTION_PATH = "./experiment_result/wrong_prediction_in_{}.txt".format(TASK3_TEST_DATA_PATH.split("/")[2][:-4])