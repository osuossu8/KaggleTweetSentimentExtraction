import os
import sys
import tokenizers

sys.path.append("/usr/src/app/kaggle/tweet-sentiment-extraction")


MAX_LEN = 192
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 50
NUM_FOLDS = 4
ROBERTA_PATH = "inputs/roberta-base"

TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{ROBERTA_PATH}/vocab.json", 
    merges_file=f"{ROBERTA_PATH}/merges.txt", 
    lowercase=True,
    add_prefix_space=True
)

INPUT_DIR = 'inputs'
OUT_DIR = 'models'
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

FOLD0_ONLY = False
DEBUG = False
