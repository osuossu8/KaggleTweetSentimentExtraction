import os
import sys
import tokenizers

sys.path.append("/usr/src/app/kaggle/tweet-sentiment-extraction")


MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 50
BERT_PATH = "inputs/bert-large-uncased-wwm-finetuned-squad"

TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(BERT_PATH, "vocab.txt"),
    lowercase=True
)

INPUT_DIR = 'inputs'
OUT_DIR = 'models'
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

FOLD0_ONLY = False
DEBUG = False
