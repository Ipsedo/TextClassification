from os.path import isdir, exists, join
import pickle as pkl
from models import ConvModelDBPedia_V1
from doc import __padding__, process_doc, pass_to_idx_and_padd
import torch as th
from tqdm import tqdm
from statistics import mean


if __name__ == "__main__":
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
