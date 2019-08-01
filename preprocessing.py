import torch as th
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from random import choice, shuffle
from tqdm import tqdm
#from yandex.Translater import Translater
#from nlpaug.augmenter.word import Word2vecAug, GloVeAug, FasttextAug
from math import ceil
from threading import Thread

__padding__ = "<padding>"
lemma = WordNetLemmatizer()
lemma.lemmatize("unused")
english_stopwords = open("./datasets/english_stopwords.txt").read().split()
#aug = Word2vecAug(model_path="/home/samuel/Documents/Stage_SG/nlpaug_models/GoogleNews-vectors-negative300.bin")
#glove_aug = GloVeAug(model_path="../../data/glove.6B.50d.txt")
#ft_aug = FasttextAug(model_path="/home/samuel/Documents/Stage_SG/nlpaug_models/wiki-news-300d-1M.vec")


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


def duplicate_class(x_list: list, y_list: list, max_per_class=300):
    classes = {}
    for c in y_list:
        classes[c] = classes[c] + 1 if c in classes else 1

    per_class_doc = {c: [] for c, _ in classes.items()}

    for x, y in zip(x_list, y_list):
        per_class_doc[y].append(x)

    x_res = x_list.copy()
    y_res = y_list.copy()

    for y, c in classes.items():
        to_add = max_per_class - c
        to_add = max(0, to_add)

        for _ in range(to_add):
            x_res.append(choice(per_class_doc[y]))
            y_res.append(y)

    tmp = list(zip(x_res, y_res))
    shuffle(tmp)
    x_res, y_res = zip(*tmp)

    return list(x_res), list(y_res)


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
            if w in vocab:
                new_s.append(vocab[w])
        to_add = max_len - len(new_s)
        new_s += [vocab[__padding__]] * to_add
        res.append(new_s)

    return th.tensor(res)


def filter_size(data, labels, limit=1000):
    tmp = list(filter(lambda t: len(t[0]) < limit, list(zip(data, labels))))

    x, y = zip(*tmp)
    return list(x), list(y)


def count_world(sentence_list):
    count = {}
    for s in tqdm(sentence_list):
        for w in s:
            count[w] = count[w] + 1 if w in count else 1
    return count


def filter_word_occ(sentence_list, word_count, limit=10):
    return [[w for w in s if word_count[w] > limit] for s in tqdm(sentence_list)]


def filter_class(sentences, labels, label_count, limit_up=10000, limit_down=0):
    sentence_res = []
    labels_res = []

    for s, l in zip(sentences, labels):
        if limit_down <= label_count[l] <= limit_up:
            sentence_res.append(s)
            labels_res.append(l)

    return sentence_res, labels_res


def limit_class_occurence(sentence_list, label_list, limit=10000):
    class_count = {}
    
    new_sentence_list = []
    new_label_list = []
    for s, l in tqdm(zip(sentence_list, label_list)):
        class_count[l] = 1 + class_count[l] if l in class_count else 1
        
        if class_count[l] <= limit:
            new_sentence_list.append(s)
            new_label_list.append(l)
    
    return new_sentence_list, new_label_list


def filter_limit_class(sentence_list, label_list, class_count, limit_up=5000, limit_down=500):
    new_sentence = []
    new_labels = []

    class_counter = {}

    for s, l in tqdm(zip(sentence_list, label_list)):

        class_counter[l] = 1 + class_counter[l] if l in class_counter else 1

        if class_count[l] >= limit_down and class_counter[l] <= limit_up:
            new_sentence.append(s)
            new_labels.append(l)

    return new_sentence, new_labels


def compute_class_weights(label_list, eps=1e-6):
    counter = {}

    for l in label_list:
        counter[l] = 1 + counter[l] if l in counter else 1

    weights = {}

    max_occurence = max(map(lambda t: t[1], counter.items()))
    total_example = len(label_list)

    for l, c in counter.items():
        weights[l] = 50.0 / c
        if weights[l] == 0:
            weights[l] = eps

    return weights


def rewrite_sentence(sentence: str):
    """word_to_change = int(len(sentence) * ratio)

    for _ in range(word_to_change):
        i = choice(range(len(sentence)))

        possible_senses = wordnet.synsets(sentence[i])

        if possible_senses:
            syn = possible_senses[0].lemmas()

            new_words = choice(syn).name().split("_")
            del sentence[i]
            sentence[i:i] = new_words

    return sentence"""
    return glove_aug.augment(sentence).replace("_", " ")


def rewrite_corpus(sentence_list, label_list, limit_augmentation=800):
    counter = {}

    for l in label_list:
        counter[l] = 1 + counter[l] if l in counter else 1

    new_sentence_list = []
    new_label_list = []
    
    to_add_per_data = {}
    for c, count in counter.items():
        to_add = int(limit_augmentation / count)
        to_add_per_data[c] = to_add

    for s, l in tqdm(zip(sentence_list, label_list)):

        if counter[l] < limit_augmentation:
            for _ in range(to_add_per_data[l]):
                rewrited_sentence = rewrite_sentence(s)

                new_sentence_list.append(rewrited_sentence)
                new_label_list.append(l)

        new_sentence_list.append(s)
        new_label_list.append(l)

    return new_sentence_list, new_label_list


def rewrite_corpus_threads(sentence_list, label_list, limit_augmentation=800):
    counter = {}

    for l in label_list:
        counter[l] = 1 + counter[l] if l in counter else 1

    new_sentence_list = []
    new_label_list = []

    to_add_per_data = {}
    for c, count in counter.items():
        to_add = int(limit_augmentation / count)
        to_add_per_data[c] = to_add

    nb_thread = 4
    nb_pool = ceil(len(sentence_list) / nb_thread)

    tuple_list = list(zip(sentence_list, label_list))

    for idx in tqdm(range(nb_pool)):

        thds = []
        i_min = idx * nb_thread
        i_max = (idx + 1) * nb_thread
        i_max = i_max if i_max < len(sentence_list) else len(sentence_list)

        def run(s, l):
            if counter[l] < limit_augmentation:
                for _ in range(to_add_per_data[l]):
                    rewrited_sentence = rewrite_sentence(s)

                    new_sentence_list.append(rewrited_sentence)
                    new_label_list.append(l)

            new_sentence_list.append(s)
            new_label_list.append(l)

        for sent, lbl in tuple_list[i_min:i_max]:
            t = Thread(target=run, args=(sent, lbl))
            t.start()
            thds.append(t)

        for t in thds:
            t.join()

    return new_sentence_list, new_label_list


def back_translate(sentence: str) -> list:
    #gs = Goslate()
    langs = ["af", "sq", "ar", "am", "zh", "ja", "ms", "mk", "ht", "vi"]
    main_lang = "en"

    new_sentence = [sentence]

    for l in langs:
        tr_1 = Translater(key="trnsl.1.1.20190715T073231Z.8560179895122412.c89f6ceed11aead6234cadbf49a04719f96027e0", from_lang=main_lang, to_lang=l, text=sentence)
        tr_2 = Translater(key="trnsl.1.1.20190715T073231Z.8560179895122412.c89f6ceed11aead6234cadbf49a04719f96027e0", from_lang=l, to_lang=main_lang, text=tr_1.translate())
        s = tr_2.translate()
        new_sentence.append(s)

    return new_sentence


def back_translate_corpus(sentence_list, label_list, limit_augmentation=800):
    counter = {}

    for l in label_list:
        counter[l] = 1 + counter[l] if l in counter else 1

    new_sentence_list = []
    new_label_list = []

    for s, l in tqdm(zip(sentence_list, label_list)):

        if counter[l] < limit_augmentation:
            generated_back_translations = back_translate(s)
            classes = [l] * len(generated_back_translations)

            new_sentence_list += generated_back_translations
            new_label_list += classes
        else:
            new_sentence_list.append(s)
            new_label_list.append(l)

    return new_sentence_list, new_label_list


def augment_main(in_file, out_file_train, out_file_dev):
    dbpedia = open(in_file).readlines()

    x = []
    y = []

    class_to_idx = {}
    class_count = {}

    for l in tqdm(dbpedia):
        lbl = l.split("|||")[0]
        txt = l.split("|||")[1]

        if lbl not in class_to_idx:
            class_to_idx[lbl] = len(class_to_idx)

        y.append(class_to_idx[lbl])
        x.append(txt)

        class_count[class_to_idx[lbl]] = 1 + class_count[class_to_idx[lbl]] if class_to_idx[lbl] in class_count else 1

    tmp = list(zip(x, y))
    shuffle(tmp)
    x, y = zip(*tmp)

    print("Nb class : %d" % len(class_to_idx))
    print("Nb abstracts : %d" % len(x))

    x, y = filter_limit_class(x, y, class_count, limit_up=1000, limit_down=100)

    idx_to_class = {idx: cl for cl, idx in class_to_idx.items()}

    class_to_idx = {}

    new_y = []
    for lbl in y:
        if idx_to_class[lbl] not in class_to_idx:
            class_to_idx[idx_to_class[lbl]] = len(class_to_idx)
        new_y.append(class_to_idx[idx_to_class[lbl]])
    y = new_y

    print("Nb class : %d" % len(class_to_idx))
    print("Nb abstracts : %d" % len(x))

    print("begin rewriting...")
    
    ratio = 0.7
    nb_train = int(len(x) * ratio)
    x_train, y_train = x[:nb_train], y[:nb_train]
    x_dev, y_dev = x[nb_train:], y[nb_train:]

    x_train, y_train = rewrite_corpus_threads(x_train, y_train, limit_augmentation=1000)

    print("Nb abstracts : %d" % len(x))

    class_count = {}
    for l in y_train:
        class_count[l] = 1 + class_count[l] if l in class_count else 1
    print(sorted(class_count.items(), key=lambda t: t[1]))

    print("writing file...")

    f = open(out_file_train, "w")

    for s, l in tqdm(zip(x_train, y_train)):
        f.write(s + "|||" + l + "\n")

    f.close()
    
    f = open(out_file_dev, "w")

    for s, l in tqdm(zip(x_dev, y_dev)):
        f.write(s + "|||" + l + "\n")

    f.close()

    
if __name__ == "__main__":
    augment_main("./datasets/dbpedia_pp_filtered.txt", "dbpedia_filtered_glove-augmented_train.txt", "dbpedia_filtered_glove-augmented_dev.txt")
