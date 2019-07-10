import torch as th
from models import ConvModelDBPedia_V1
import pickle as pkl
from doc import __padding__, process_doc, pass_to_idx_and_padd


def test():
    model_file_name = "./saved/CNN-V1_128-184-256.model"
    vocab_file_name = "./saved/vocab.pkl"
    class_to_idx_file_name = "./saved/class_to_idx.pkl"

    test_doc = "/home/samuel/Documents/Stage_SG/xlnet.txt"

    vocab = pkl.load(open(vocab_file_name, "rb"))
    class_to_idx = pkl.load(open(class_to_idx_file_name, "rb"))
    idx_to_class = {idx: cl for cl, idx in class_to_idx.items()}

    max_len = 499

    model = ConvModelDBPedia_V1(len(vocab), len(class_to_idx), vocab[__padding__])
    model.load_state_dict(th.load(model_file_name))

    model.eval()

    doc = open(test_doc, "r").read()
    doc = process_doc([doc])
    doc = pass_to_idx_and_padd(doc, vocab, max_len)

    out = model(doc)

    predicted = out.argmax(dim=-1).view(-1).item()

    print("Predicted : %s with %f probability" % (idx_to_class[predicted], out.view(-1)[predicted].item()))


if __name__ == "__main__":
    test()
