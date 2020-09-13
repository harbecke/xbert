import numpy as np


def calculate_correlation(relevance_dict_1, relevance_dict2):
    zipped_value_list = []
    for word_key, relevance in relevance_dict_1.items():
        zipped_value_list.append([relevance, relevance_dict2[word_key]])
    return np.corrcoef(np.array(zipped_value_list), rowvar=False)[0][1]


def confusion_matrix(instances_dict, threshold=0.5):
    cm = [[0, 0], [0, 0]]

    for key, values in instances_dict.items():
        if values[2] == '0':
            idx1 = 0
            if values[1] > threshold:
                idx2 = 0
            else:
                idx2 = 1
        elif values[2] == '1':
            idx1 = 1
            if values[1] > 1 - threshold:
                idx2 = 1
            else:
                idx2 = 0
        cm[idx1][idx2] += 1

    return cm


def phi_coefficient(confusion_matrix):
    num = confusion_matrix[0][0]*confusion_matrix[1][1] - confusion_matrix[0][1]*confusion_matrix[1][0]
    den = (confusion_matrix[0][0]+confusion_matrix[0][1]) * (confusion_matrix[0][0]+confusion_matrix[1][0]) * \
          (confusion_matrix[1][1]+confusion_matrix[0][1]) * (confusion_matrix[1][1]+confusion_matrix[1][0])
    return num / den**0.5


def relevances_lists_to_t_values(relevance_lists):
    cav = [(len(relevances), np.average(relevances), np.var(relevances)) for relevances in relevance_lists]
    num = cav[0][1] - cav[1][1]
    den = cav[0][2]/cav[0][0] + cav[1][2]/cav[1][0]
    dgf_den = cav[0][2]**2/cav[0][0]**3 + cav[1][2]**2/cav[1][0]**3
    return num / den**0.5, den**2 / dgf_den
