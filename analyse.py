from os.path import isdir, exists, join
import pickle as pkl
from models import ConvModelDBPedia_V1
from preprocessing import __padding__
from preprocessing import *
import torch as th
from tqdm import tqdm
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np


def vocab_count():
    dbpedia = open("./datasets/dbpedia_pp_filtered.txt").readlines()

    x = []
    y = []

    for l in tqdm(dbpedia):
        txt = l.split("|||")[1]
        lbl = l.split("|||")[0]
        x.append(txt)
        y.append(lbl)

    print("Nb abstracts : %d" % len(x))

    x = process_doc(x)

    word_count = count_world(x)

    x = filter_word_occ(x, word_count, 10)

    x, y = filter_size(x, y, 500)

    vocab = create_vocab(x)
    print("Vocab created (size = %d)" % len(vocab))


def visu_dbpedia():
    dbpedia = open("../../data/dbpedia_pp_simplified-1.txt").readlines()

    x = []
    y = []

    class_to_idx = {}
    class_count = {}

    for l in tqdm(dbpedia):
        lbl = l.split("|||")[0]
        txt = l.split("|||")[1]

        if lbl not in class_to_idx:
            class_to_idx[lbl] = len(class_to_idx)
        class_count[lbl] = 1 + class_count[lbl] if lbl in class_count else 1

        y.append(class_to_idx[lbl])
        x.append(txt)

    print(len(class_to_idx))
    print(class_count)

    count = np.asarray([c for _, c in class_count.items()])
    type_name = np.asarray([t for t, _ in class_count.items()])

    mask = count > 1000
    count = count[mask]
    type_name = type_name[mask]

    plt.bar(type_name, count)
    plt.xticks(rotation="vertical")
    plt.show()

def main_2():
    path = "./results/results_cnn-v1-128-184-256_limited-10000"

    if not (exists(path) and isdir(path)):
        print("You must extract the zip file \"results_cnn-v1-128-184-256_limited-10000.zip\" !")
        exit(0)

    model_file_name = join(path, "CNN-V1_128-184-256.model")
    vocab_file_name = join(path, "vocab.pkl")
    class_to_idx_file_name = join(path, "class_to_idx.pkl")

    vocab = pkl.load(open(vocab_file_name, "rb"))
    class_to_idx = pkl.load(open(class_to_idx_file_name, "rb"))
    idx_to_class = {idx: cl for cl, idx in class_to_idx.items()}

    max_len = 499

    model = ConvModelDBPedia_V1(len(vocab), len(class_to_idx), vocab[__padding__])
    model.load_state_dict(th.load(model_file_name))

    model.eval()

    dbpedia = open("./datasets/dbpedia_pp_filtered.txt").readlines()

    class_count = {}

    x = []
    y = []

    for l in tqdm(dbpedia):
        lbl = l.split("|||")[0]
        txt = l.split("|||")[1]

        y.append(class_to_idx[lbl])
        x.append(txt)

        class_count[lbl] = 1 + class_count[lbl] if lbl in class_count else 1

    print(sorted(class_count.items(), key=lambda t: t[1], reverse=True))
    print("Moyenne nombre d'occurrence par classe : %f" % mean(map(lambda t: t[1], class_count.items())))

    plt.bar(list(range(len(class_count))),
            np.log10(list(map(lambda t: t[1], sorted(class_count.items(), key=lambda t: t[1], reverse=True)))))
    plt.ylabel("log10 - Effectif")
    plt.xlabel("Class ID")
    plt.title("DBPedia : Effectifs classes (log10)")
    plt.legend()
    plt.show()

    plt.bar(list(range(len(class_count))),
            list(map(lambda t: t[1], sorted(class_count.items(), key=lambda t: t[1], reverse=True))))
    plt.ylabel("Effectif")
    plt.xlabel("Class ID")
    plt.title("DBPedia : Effectifs classes")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    vocab_count()
