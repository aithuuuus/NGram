import numpy as np

def load(corpus, tokenizer):
    '''
    load and prepare the training dataset
    return: size of vocabularies, dict of token to vocab, tokenized data
    '''
    # load the corpus
    with open(f'data/{corpus}', 'r') as f:
        corpus = f.read()

    # tokenization
    if tokenizer == 'character':
        corpus = list(corpus)
        vocabs = sorted(list(set(corpus)))
        v_size = len(vocabs)
        v2t_map = dict(zip(vocabs, range(v_size)))
        t2v_map = dict(zip(range(v_size), vocabs))
        corpus = [v2t_map[i] for i in corpus]
    elif tokenizer == 'bpe':
        raise NotImplementedError
    else:
        raise NotImplementedError(f"Incorrect tokenizer method: {tokenizer}")

    return v_size, t2v_map, corpus

class Dataset:
    def __init__(self, corpus):
        self.corpus = corpus
        if not isinstance(self.corpus, np.ndarray):
            self.corpus = np.array(self.corpus)

    def sample(self, batch_size=32, max_len=32):
        idx = np.random.randint(0, len(self.corpus)-max_len, batch_size)
        idx = np.arange(max_len)[None, :].repeat(batch_size, 0) + idx[:, None]
        return self.corpus[idx]

    def __len__(self):
        return len(self.corpus)
