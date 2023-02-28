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
        corpus = corpus.split()
        vocabs = sorted(list(set(corpus)))
        v_size = len(vocabs)
        v2t_map = dict(zip(range(v_size), vocabs))
        t2v_map = dict(zip(vocabs, range(v_size)))
        corpus = [v2t_map[i] for i in corpus]
    elif tokenizer = 'bpe':
        raise NotImplementedError

    return v_size, t2v_map, corpus

class Dataset:
    def __init__(self):
        pass

    def sample(self):
        pass
