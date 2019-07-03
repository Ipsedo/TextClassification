from tqdm import tqdm
import re


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


def main(type_file, abstract_file, out_file):
    instances_types = open(type_file).readlines()[1:]
    labels = [l.split(" ")[2].replace("<", "").replace(">", "").split("/")[-1] for l in instances_types]
    titles = [l.split(" ")[0].replace("<", "").replace(">", "").split("/")[-1] for l in instances_types]

    titles, labels = filter_things(titles, labels)

    title_to_type = create_types_dict(titles, labels)

    print(len(title_to_type))

    long_abstracts = open(abstract_file).readlines()[1:]

    dataset = create_dataset(title_to_type, long_abstracts)

    write_dataset(dataset, out_file)


if __name__ == "__main__":
    main("./datasets/instance_types_en.ttl", "./datasets/long_abstracts_en.ttl", "./datasets/dbpedia_pp.txt")
