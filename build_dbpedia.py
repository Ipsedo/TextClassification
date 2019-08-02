from tqdm import tqdm
import re
from random import shuffle


def filter_things(titles, labels):
    new_titles = []
    new_labels = []

    for t, l in tqdm(zip(titles, labels)):
        if l != "owl#Thing":
            new_labels.append(l)
            new_titles.append(t)
    return new_titles, new_labels


def create_types_dict(titles, labels):
    return {t: l for t, l in zip(titles, labels)}


def create_dataset(title_to_type, abstracts):
    data_set = []

    for line in tqdm(abstracts):
        title = line.split(" ")[0].replace("<", "").replace(">", "").split("/")[-1]
        try:
            abst = re.search("\"(.+)\"", line).group(1)
        except:
            # Prevent empty abstracts
            continue

        if title in title_to_type:
            data = (title_to_type[title], abst)
            data_set.append(data)

    return data_set


def write_dataset(data_set, file_name):
    out_file = open(file_name, "w")

    for lbl, data in tqdm(data_set):
        line = lbl + "|||" + data
        out_file.write(line + "\n")


def pass_to_parent_class(title_to_type, ontology):
    #return {t: str(ontology[l].is_a[0]).split(".")[-1] for t, l in title_to_type.items() if ontology[l] is not None}
    res = {}

    for t, l in title_to_type.items():
        if ontology[l] is not None:
            parent_type = str(ontology[l].is_a[0]).split(".")[-1]
            parent_type = l if parent_type == "Thing" else parent_type
            res[t] = parent_type

    return res


def main(type_file, abstract_file, out_file):
    """

    :param type_file:
    :param abstract_file:
    :param out_file:
    :return:
    """
    instances_types = open(type_file).readlines()[1:]
    instances_types = [l for l in instances_types if "ontology" in l]

    labels = [re.findall(r"[A-Za-z]+", l.split(" ")[2].replace("<", "").replace(">", ""))[-1] for l in instances_types]
    titles = [re.findall(r"[A-Za-z]+", l.split(" ")[0].replace("<", "").replace(">", ""))[-1] for l in instances_types]

    titles, labels = filter_things(titles, labels)

    title_to_type = create_types_dict(titles, labels)

    long_abstracts = open(abstract_file).readlines()[1:]

    dataset = create_dataset(title_to_type, long_abstracts)

    write_dataset(dataset, out_file)
    

def shuffle_data(file_name: str, out_file_name: str):
    file = open(file_name, "r")
    lines = file.readlines()
    shuffle(lines)
    
    out_file = open(out_file_name, "w")
    
    for l in tqdm(lines):
        out_file.write(l)
    
    file.close()
    out_file.close()

if __name__ == "__main__":
    main("./datasets/instance_types_en.ttl",
         "./datasets/long_abstracts_en.ttl",
         "./datasets/dbpedia_pp_filtered.txt")
