import nltk
from nltk.corpus import reuters
from doc import *
from doc import __padding__
from models import ConvModelReuters, ConvModelWiki, ConvModelDBPedia_V1, ConvModelDBPedia_V2
import torch.nn as nn
from math import ceil
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchnet.meter import AUCMeter
import gc, sys
from pympler.tracker import SummaryTracker


nltk.download('reuters')


def reuters():
    classes = ["earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "wheat", "ship", "corn"]

    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_docs = []
    train_labels = []
    dev_docs = []
    dev_labels = []

    for doc_id in reuters.fileids():
        if doc_id.startswith("train"):
            train_docs.append(reuters.raw(doc_id))
            train_labels.append(reuters.categories(doc_id)[0])
        else:
            dev_docs.append(reuters.raw(doc_id))
            dev_labels.append(reuters.categories(doc_id)[0])

    x_train = []
    y_train = []

    for doc, cat in zip(train_docs, train_labels):
        if cat in class_to_idx:
            x_train.append(doc)
            y_train.append(class_to_idx[cat])

    x_dev = []
    y_dev = []

    for doc, cat in zip(dev_docs, dev_labels):
        if cat in class_to_idx:
            x_dev.append(doc)
            y_dev.append(class_to_idx[cat])

    x_train, y_train = duplicate_class(x_train, y_train)

    x_train = process_doc(x_train)
    x_dev = process_doc(x_dev)

    y_train = th.tensor(y_train)
    y_dev = th.tensor(y_dev)

    max_len = get_sentence_max_len(x_train + x_dev)

    vocab = create_vocab(x_train + x_dev)

    x_train = pad_sentence(x_train, max_len)
    x_dev = pad_sentence(x_dev, max_len)

    x_train = pass_to_idx(x_train, vocab)
    x_dev = pass_to_idx(x_dev, vocab)

    print(x_train.size())

    m = ConvModelReuters(len(vocab), len(class_to_idx), max_len, vocab[__padding__])
    loss_fn = nn.NLLLoss()

    m.cuda()
    loss_fn.cuda()

    optim = th.optim.SGD(m.parameters(), lr=1e-3)

    batch_size = 4
    nb_batch = ceil(x_train.size(0) / batch_size)

    nb_epoch = 40

    losses = []
    acc = []

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

            loss = loss_fn(out, y_b)

            loss.backward()
            optim.step()

            sum_loss += loss.cpu().item()

        print("Epoch %d, loss = %f" % (e, sum_loss / nb_batch))

        losses.append(sum_loss / nb_batch)

        with th.no_grad():
            m.eval()

            nb_batch_test = ceil(x_dev.size(0) / batch_size)

            correct = 0

            aucs = {i: AUCMeter() for i in range(len(classes))}

            for i in tqdm(range(nb_batch_test)):
                i_min = i * batch_size
                i_max = (i + 1) * batch_size
                i_max = i_max if i_max < x_dev.size(0) else x_dev.size(0)

                x_b, y_b = x_dev[i_min:i_max].cuda(), y_dev[i_min:i_max].cuda()

                out = m(x_b)

                correct += th.where(out.argmax(dim=1) == y_b, th.tensor(1).cuda(), th.tensor(0).cuda()).sum().cpu().item()

                y_b_one_hot = th.eye(len(classes))[y_b]

                for _, idx in class_to_idx.items():
                    aucs[idx].add(out[:, idx], y_b_one_hot[:, idx])

            print("Test : correct = %d / %d, %f" % (correct, x_dev.size(0), correct / x_dev.size(0)))
            print((y_dev == class_to_idx["earn"]).sum().item())
            for i in range(len(classes)):
                print("AUC %s = %f" % (idx_to_class[i], aucs[i].value()[0]))
            acc.append(correct / x_dev.size(0))

    plt.plot(losses, 'r')
    plt.plot(acc, "b")
    plt.xlabel("Epoch")
    plt.show()


def wiki():
    # TODO add script for Wiki dump with LDA
    wiki = open("/home/samuel/Documents/Stage_SG/lda/labelised_raw_wiki.txt").readlines()
    pos_class = [89]
    neg_class = [75]

    x = []
    y = []

    for l in wiki:
        splitted = l.split("|||")
        lbl = int(splitted[0])
        txt = splitted[2]

        if lbl in pos_class or lbl in neg_class:
            lbl = 1 if lbl in pos_class else 0

            x.append(txt)
            y.append(lbl)

    tmp = list(zip(x, y))
    shuffle(tmp)
    x, y = zip(*tmp)
    print(len(x))

    ratio = 0.7
    nb_train = int(len(x) * ratio)
    x_train, y_train = x[:nb_train], y[:nb_train]
    x_dev, y_dev = x[nb_train:], y[nb_train:]

    x_train = process_doc(x_train)
    x_dev = process_doc(x_dev)

    y_train = th.tensor(y_train).to(th.float)
    y_dev = th.tensor(y_dev).to(th.float)

    max_len = get_sentence_max_len(x_train + x_dev)

    vocab = create_vocab(x_train + x_dev)

    x_train = pad_sentence(x_train, max_len)
    x_dev = pad_sentence(x_dev, max_len)

    x_train = pass_to_idx(x_train, vocab)
    x_dev = pass_to_idx(x_dev, vocab)

    print(x_train.size(), y_train.size())
    print(x_dev.size(), y_dev.size())

    batch_size = 4
    nb_batch = ceil(x_train.size(0) / batch_size)

    nb_epoch = 10

    m = ConvModelWiki(len(vocab), max_len, vocab[__padding__])
    loss_fn = nn.MSELoss()

    m.cuda()
    loss_fn.cuda()

    optim = th.optim.SGD(m.parameters(), lr=1e-4)

    losses = []
    acc = []
    aucs = []

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

            loss = loss_fn(out, y_b)

            loss.backward()
            optim.step()

            sum_loss += loss.cpu().item()

        print("Epoch %d, loss = %f" % (e, sum_loss / nb_batch))

        losses.append(sum_loss / nb_batch)

        with th.no_grad():
            m.eval()

            nb_batch_test = ceil(x_dev.size(0) / batch_size)

            auc = AUCMeter()

            for i in tqdm(range(nb_batch_test)):
                i_min = i * batch_size
                i_max = (i + 1) * batch_size
                i_max = i_max if i_max < x_dev.size(0) else x_dev.size(0)

                x_b, y_b = x_dev[i_min:i_max].cuda(), y_dev[i_min:i_max].cuda()

                out = m(x_b)

                auc.add(out, y_b)
            auc_val = auc.value()[0]

            aucs.append(auc_val)

            print("Test, AUC ROC = %f" % (auc_val,))

    plt.plot(losses, "r", label="loss value")
    plt.plot(aucs, "b", label="roc auc value")
    plt.xlabel("Epoch")
    plt.title("Wiki labeled with LDA : CNN perf (topic {} vs {})".format(pos_class, neg_class))
    plt.legend()
    plt.show()


def dbpedia():
    tracker = SummaryTracker()

    dbpedia = open("./datasets/dbpedia_pp.txt").readlines()

    x = []
    y = []

    class_to_idx = {}

    for l in tqdm(dbpedia):
        lbl = l.split("|||")[0]
        txt = l.split("|||")[1]

        if lbl not in class_to_idx:
            class_to_idx[lbl] = len(class_to_idx)

        y.append(class_to_idx[lbl])
        x.append(txt)

    tmp = list(zip(x, y))
    shuffle(tmp)
    x, y = zip(*tmp)

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

    tracker.print_diff()
    a = input("A : ")
    exit()

    batch_size = 16
    nb_batch = ceil(x_train.size(0) / batch_size)

    nb_epoch = 10

    m = ConvModelDBPedia_V1(len(vocab), len(class_to_idx), vocab[__padding__])
    loss_fn = nn.NLLLoss()

    m.cuda()
    loss_fn.cuda()

    optim = th.optim.SGD(m.parameters(), lr=1e-4)

    losses = []
    acc = []

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

                x_b, y_b = x_dev[i_min:i_max].cuda(), y_dev[i_min:i_max].cuda()

                out = m(x_b)

                correct_answer += (out.argmax(dim=-1) == y_b.to(th.long)).sum().item()

        correct_answer /= x_dev.size(0)

        acc.append(correct_answer)
        print("Accuracy = %f" % correct_answer)

    plt.plot(losses, "r", label="loss value")
    plt.plot(acc, "b", label="accuracy value")
    plt.xlabel("Epoch")
    plt.title("DBPedia (onthology classification)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #reuters()
    #wiki()
    dbpedia()
