import csv
from collections import defaultdict


def create_candidate_dicts(candidate_instances, candidate_results):
    id_counter = -1
    replace_counter = 0
    weight_sum = 0
    candidates_dict = defaultdict(lambda: defaultdict(list))
    instances_dict = defaultdict(tuple)
    for instance, result in zip(candidate_instances, candidate_results):
        if instance.id == id_counter:
            if weight_sum == 100.0:
                weight_sum = 0
                replace_counter += 1
            candidates_dict[id_counter][replace_counter].append(
                (instance.token_fields["sent"].tokens, instance.weight, result))
            weight_sum += instance.weight

        else:
            replace_counter = 0
            weight_sum = 0
            id_counter += 1
            instances_dict[id_counter] = [instance.token_fields["sent"].tokens, result]

    return instances_dict, candidates_dict


def read_instances_dict_and_append_label_cola(data1, instances_dict):
    with open(data1, 'r') as csv_file:
        spamreader = csv.reader(csv_file, delimiter='\t')
        next(spamreader)
        for idx1, line in enumerate(spamreader):
            instances_dict[idx1].append(line[1])
    return


def read_instances_dict_and_append_label_fava(data1, instances_dict):
    with open(data1, 'r') as csv_file:
        spamreader = csv.reader(csv_file, delimiter='\t')
        next(spamreader)
        for idx1, line in enumerate(spamreader):
            instances_dict[idx1].extend([line[1], line[0]])
    return


def filter_and_sort_candidates(candidates_dict, min_weight=2):
    for index, values in candidates_dict.items():
        values = filter(lambda x: x[1] >= min_weight, values)
        candidates_dict[index] = sorted(values, reverse=True, key=lambda x: x[1])
    return


def processed_candidates_dict_to_list(candidates_dict):
    return [[(sentence_list[index], weight, prediction) for sentence_list, weight, prediction in values]
            for index, values in candidates_dict.items()]
