import numpy as np


def calculate_correlation(relevance_dict_1, relevance_dict2):
    zipped_value_list = []
    for word_key, relevance in relevance_dict_1.items():
        zipped_value_list.append([relevance, relevance_dict2[word_key]])
    return np.corrcoef(np.array(zipped_value_list), rowvar=False)[0][1]
