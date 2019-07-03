import torch as th
import re
from nltk.stem import WordNetLemmatizer
from random import choice, shuffle
from tqdm import tqdm

__padding__ = "<padding>"
lemma = WordNetLemmatizer()
lemma.lemmatize("unused")
english_stopwords = open("./datasets/english_stopwords.txt").read().split()


def split_doc(s):
    return re.findall(r"[A-Za-z]+", s)


def to_lower(word_list: list) -> list:
    return [w.lower() for w in word_list]


def filter_words(word_list: list, to_filter: list) -> list:
    return [w for w in word_list if w not in to_filter]


def lemma_words(word_list: list) -> list:
    return [lemma.lemmatize(w) for w in word_list]


def process_doc(sentence_list: list) -> list:
    res = []
    for text in tqdm(sentence_list):
        l = split_doc(text)
        l = to_lower(l)
        #l = filter_words(l, english_stopwords)
        #l = lemma_words(l)
        res.append(l)
    return res


def duplicate_class(x_list: list, y_list: list):
    classes = {}
    for c in y_list:
        classes[c] = classes[c] + 1 if c in classes else 1

    per_class_doc = {c: [] for c, _ in classes.items()}

    for x, y in zip(x_list, y_list):
        per_class_doc[y].append(x)

    x_res = x_list.copy()
    y_res = y_list.copy()

    m = max(map(lambda t: t[1], classes.items()))

    for y, c in classes.items():
        to_add = m - c
        #to_add = min(to_add, 300)

        for _ in range(to_add):
            x_res.append(choice(per_class_doc[y]))
            y_res.append(y)

    tmp = list(zip(x_res, y_res))
    shuffle(tmp)
    x_res, y_res = zip(*tmp)

    return x_res, y_res


def create_vocab(sentence_list):
    vocab = {__padding__: 0}
    for s in sentence_list:
        for w in s:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def get_sentence_max_len(sentence_list):
    return max(map(lambda s: len(s), sentence_list))


def pad_sentence(sentence_list, max_len):
    res = []
    for s in sentence_list:
        to_add = max_len - len(s)
        res.append(s + [__padding__] * to_add)
    return res


def pass_to_idx(sentence_list, vocab):
    res = []

    for s in sentence_list:
        new_s = []
        for w in s:
            new_s.append(vocab[w])
        res.append(new_s)

    return th.tensor(res)


def pass_to_idx_and_padd(sentence_list, vocab, max_len):
    res = []

    for s in tqdm(sentence_list):
        new_s = []
        for w in s:
            new_s.append(vocab[w])
        to_add = max_len - len(new_s)
        new_s += [vocab[__padding__]] * to_add
        res.append(new_s)

    return th.tensor(res)


def filter_size(data, labels, limit=1000):
    tmp = list(filter(lambda t: len(t[0]) < limit, list(zip(data, labels))))

    return zip(*tmp)


def count_world(sentence_list):
    count = {}
    for s in sentence_list:
        for w in s:
            count[w] = count[w] + 1 if w in count else 1
    return count


def filter_word_occ(sentence_list, word_count, limit=10):
    return [[w for w in s if word_count[w] > limit] for s in sentence_list]


