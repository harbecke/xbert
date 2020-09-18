import pandas as pd


def read_fava_dataset(dataset_file):
    df = pd.read_csv(dataset_file, delimiter="\t", usecols=[3])
    return [sentence.split(" ") for sentence in df["inputs"].tolist()]


def read_verb_list(verb_list_file):
    df = pd.read_csv(verb_list_file)
    return df["verb"].tolist()


def token_lists_to_verb_idx():
    list_of_list_of_tokens = read_fava_dataset("data/fava/dev.tsv")
    verb_list = read_verb_list("data/fava/only_verbs.csv")

    output_list = []
    for sentence_list in list_of_list_of_tokens:
        for idx, token in enumerate(sentence_list):
            if token in verb_list:
                output_list.append(idx)
                break
    return output_list


if __name__ == "__main__":
    idx_list = token_lists_to_verb_idx()
    with open('data/fava/verb_idx.txt', 'w') as f:
        for item in idx_list:
            f.write("%s\n" % item)
