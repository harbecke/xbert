import os
import dill
from collections import defaultdict


def experiment_load_relevances(experiment_dir: str,
                               relevance_filename: str = "relevances.pkl"):

    experiment_relevances = {}
    for subfolder in os.listdir(experiment_dir):
        relevance_file = os.path.join(experiment_dir, subfolder, relevance_filename)
        with open(relevance_file, "rb") as f:
            relevances = dill.load(f)
            experiment_relevances[subfolder] = relevances

    return experiment_relevances


def relevances_to_lists(instances_dict, relevances_dict, method, threshold=0.9):
    assert instances_dict.keys() == relevances_dict.keys()
    relevance_lists = [[], []]

    for key, values in instances_dict.items():
        if values[1] > threshold:
            relevance_lists[int(values[2])].append(method(relevances_dict[key].values()))

    return relevance_lists


def extract_verb_relevances(relevances_dict, index_list):
    assert len(relevances_dict.keys()) == len(index_list)
    verb_relevance_dict = defaultdict(lambda: dict())

    for key, values in relevances_dict.items():
        verb_relevance_dict[key][0] = list(values.values())[index_list[key]]

    return verb_relevance_dict
