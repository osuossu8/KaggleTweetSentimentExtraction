import os
import sys
import tokenizers

sys.path.append("/usr/src/app/kaggle/tweet-sentiment-extraction")
sys.path.insert(0, "inputs/sentencepiece-pb2/")


import sentencepiece as spm
import sentencepiece_pb2


MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 50
NUM_FOLDS = 5
ROBERTA_PATH = "inputs/roberta-base"

ALBERT_PATH = "inputs/albert-large-v1"
SP_PB2_PATH = "inputs/sentencepiece-pb2"


class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(model_path, 'spiece.model'))
        
    def encode(self, sentence):
        spt = sentencepiece_pb2.SentencePieceText()
        spt.ParseFromString(self.sp.encode_as_serialized_proto(sentence))
        offsets = []
        tokens = []
        for piece in spt.pieces:
            tokens.append(piece.id)
            offsets.append((piece.begin, piece.end))
        return tokens, offsets


#TOKENIZER = tokenizers.ByteLevelBPETokenizer(
#    vocab_file=f"{ROBERTA_PATH}/vocab.json", 
#    merges_file=f"{ROBERTA_PATH}/merges.txt", 
#    lowercase=True,
#    add_prefix_space=True
#)

TOKENIZER = SentencePieceTokenizer(ALBERT_PATH)

INPUT_DIR = 'inputs'
OUT_DIR = 'models'
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

FOLD0_ONLY = False
DEBUG = False


if __name__ == '__main__':
    print(TOKENIZER.encode('hi, how are you?'))
