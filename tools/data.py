import numpy as np

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def load(corpus, tokenizer):
    '''
    load and prepare the training dataset
    return: size of vocabularies, dict of token to vocab, tokenized data
    '''
    # load the corpus
    file = f'data/{corpus}'
    with open(file, 'r') as f:
        corpus = f.read()

    # tokenization
    if tokenizer == 'character':
        corpus = list(corpus)
        vocabs = sorted(list(set(corpus)))
        v_size = len(vocabs)
        v2t_map = dict(zip(vocabs, range(v_size)))
        t2v_map = dict(zip(range(v_size), vocabs))
        corpus = [v2t_map[i] for i in corpus]
        return v_size, t2v_map, corpus
    elif tokenizer == 'bpe':
        # we directly use the tokenizer from transformers
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train([file], trainer)
        corpus = tokenizer.encode(corpus)
        return tokenizer.get_vocab_size(), tokenizer, corpus.ids
    else:
        raise NotImplementedError(f"Incorrect tokenizer method: {tokenizer}")


class Dataset:
    def __init__(self, corpus, batch_size, max_len):
        self.corpus = corpus
        if not isinstance(self.corpus, np.ndarray):
            self.corpus = np.array(self.corpus)
        self.cnt = 0
        self.len = len(corpus)
        self.max_len = max_len
        self.batch_size = batch_size

    def sample(self):
        # random sample
        idx = np.random.randint(0, len(self.corpus)-self.max_len, self.batch_size)
        idx = np.arange(self.max_len)[None, :].repeat(self.batch_size, 0) + idx[:, None]
        return self.corpus[idx]

    def __len__(self):
        return self.len

    def __iter__(self):
        self.cnt = 0
        return self

    def __next__(self):
        # TODO change to iterate all over the dataset
        # print(f"CNT: {self.cnt}, MAX: {int(len(self)/32)}")
        if self.cnt < int(len(self)/32):
            self.cnt += 1
            return self.sample()
        else:
            raise StopIteration
