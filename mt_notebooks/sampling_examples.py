import os
import dill

from mt_notebooks.read_instances import create_candidate_dicts


def filter_and_sort_candidates(candidates_dict, min_weight=2):
    for index, values in candidates_dict.items():
        values = filter(lambda x: x[1] >= min_weight, values)
        candidates_dict[index] = sorted(values, reverse=True, key=lambda x: x[1])
    return


def processed_candidates_dict_to_list(candidates_dict):
    return [[(sentence_list[index], weight, prediction) for sentence_list, weight, prediction in values]
            for index, values in candidates_dict.items()]


def latex_table_row(candidates_row):
    return " & ".join([f"{x[0]} & {int(x[1])} & {x[2]:.2g}" if x else " & & " for x in candidates_row])


def latex_table_double_row(candidates_row):
    return " & ".join([f"{x[0]}" if x else " " for x in candidates_row]) + " \\\\\n" + \
           " & ".join([f"({int(x[1])}, {x[2]:.2g})" if x else " " for x in candidates_row])


def processed_list_to_latex_table(candidates_list):
    transposed_list = [[x[idx] if len(x) > idx else None for x in candidates_list] for idx in
                       range(len(max(candidates_list, key=len)))]
    return " \\\\ \\hline\n".join([latex_table_double_row(transposed_row) for transposed_row in transposed_list])


def run(results_dir, index):
    with open(os.path.join(results_dir, "candidate_instances.pkl"), "rb") as in_f:
        candidate_instances, candidate_results = dill.load(in_f)
    instances_dict, candidates_dict = create_candidate_dicts(candidate_instances, candidate_results)
    filter_and_sort_candidates(candidates_dict[index])
    candidates_list = processed_candidates_dict_to_list(candidates_dict[index])
    return processed_list_to_latex_table(candidates_list)


if __name__ == "__main__":
    print(run(results_dir="../results/sst2/resampling/", index=667))
