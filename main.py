from doc import *
from doc import __padding__
from models import ConvModelDBPedia_V1, ConvModelDBPedia_V2
import torch.nn as nn
from math import ceil, floor
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchnet.meter import ConfusionMeter
import numpy as np
import pickle as pkl


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


def dbpedia():
    dbpedia = open("./datasets/dbpedia_pp_filtered.txt").readlines()

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

    x, y = filter_class(x, y, class_count)

    """idx_to_class = {idx: cl for cl, idx in class_to_idx.items()}

    class_to_idx = {}

    new_y = []
    for lbl in y:
        if idx_to_class[lbl] not in class_to_idx:
            class_to_idx[idx_to_class[lbl]] = len(class_to_idx)
        new_y.append(class_to_idx[idx_to_class[lbl]])
    y = new_y"""
    
    x, y = limit_class_occurence(x, y, limit=3000)
    
    print("Nb class : %d" % len(class_to_idx))
    print("Nb abstracts : %d" % len(x))

    x = process_doc(x)

    word_count = count_world(x)

    x = filter_word_occ(x, word_count, 10)

    x, y = filter_size(x, y, 500)

    print("Nb abstracts (filtered) : %d" % len(x))

    ratio = 0.7
    nb_train = int(len(x) * ratio)
    x_train, y_train = x[:nb_train], y[:nb_train]
    x_dev, y_dev = x[nb_train:], y[nb_train:]

    y_train = th.tensor(y_train).to(th.float)
    y_dev = th.tensor(y_dev).to(th.float)

    max_len = get_sentence_max_len(x_train + x_dev)
    print("Max len = %d" % max_len)

    vocab = create_vocab(x_train + x_dev)
    print("Vocab created (size = %d)" % len(vocab))

    x_train = pass_to_idx_and_padd(x_train, vocab, max_len)
    x_dev = pass_to_idx_and_padd(x_dev, vocab, max_len)
    print("Data passed to idx")

    print(x_train.size(), y_train.size())
    print(x_dev.size(), y_dev.size())

    batch_size = 16
    nb_batch = floor(x_train.size(0) / batch_size)

    nb_epoch = 20

    m = ConvModelDBPedia_V1(len(vocab), len(class_to_idx), vocab[__padding__])
    loss_fn = nn.NLLLoss()

    m.cuda()
    loss_fn.cuda()

    optim = th.optim.Adam(m.parameters(), lr=1e-5)

    losses = []
    acc = []

    conf_meter = ConfusionMeter(len(class_to_idx), normalized=True)

    for e in range(nb_epoch):

        m.train()

        sum_loss = 0

        for i in tqdm(range(nb_batch)):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            i_max = i_max if i_max < x_train.size(0) else x_train.size(0)

            x_b, y_b = x_train[i_min:i_max].cuda(), y_train[i_min:i_max].cuda()

            optim.zero_grad()

            out = m(x_b)

            loss = loss_fn(out, y_b.to(th.long))

            loss.backward()
            optim.step()

            sum_loss += loss.cpu().item()

        print("Epoch %d, loss = %f" % (e, sum_loss / nb_batch))

        losses.append(sum_loss / nb_batch)

        correct_answer = 0

        with th.no_grad():
            m.eval()

            nb_batch_test = ceil(x_dev.size(0) / batch_size)

            for i in tqdm(range(nb_batch_test)):
                i_min = i * batch_size
                i_max = (i + 1) * batch_size
                i_max = i_max if i_max < x_dev.size(0) else x_dev.size(0)

                x_b, y_b = x_dev[i_min:i_max].cuda(), y_dev[i_min:i_max].cuda().to(th.long)

                out = m(x_b)

                conf_meter.add(out, y_b)

                correct_answer += (out.argmax(dim=-1) == y_b).sum().item()

        correct_answer /= x_dev.size(0)

        acc.append(correct_answer)
        print("Accuracy = %f" % correct_answer)

        conf_mat = conf_meter.value()
        
        plt.figure()
        plt.matshow(conf_mat)
        plt.colorbar()
        plt.title("Confusion Matrix - Epoch %d" % e)
        plt.legend()
        plt.savefig("./results/conf_mat/conf_at_epoch-%d.png" % e)

    plt.figure()
    plt.plot(losses, "r", label="loss value")
    plt.plot(acc, "b", label="accuracy value")
    plt.xlabel("Epoch")
    plt.title("DBPedia (onthology classification)")
    plt.legend()
    plt.savefig("./results/conf_mat/accuracy.png")

    model_file_name = "CNN-V1_128-184-256.model"
    vocab_pickle_file_name = "vocab.pkl"
    class_to_idx_pickle_file_name = "class_to_idx.pkl"

    th.save(m.state_dict(), model_file_name)
    pkl.dump(vocab, open(vocab_pickle_file_name, "wb"))
    pkl.dump(class_to_idx, open(class_to_idx_pickle_file_name, "wb"))

if __name__ == "__main__":
    #reuters()
    #wiki()
    dbpedia()
    #visu_dbpedia()
