import numpy as np


def latex_table_row(candidates_row):
    return " & ".join([f"{x[0]} & {int(x[1])} & {x[2]:.2g}" if x else " & & " for x in candidates_row])


def latex_table_double_row(candidates_row):
    return " & ".join([f"{x[0]}" if x else " " for x in candidates_row]) + " \\\\\n" + \
           " & ".join([f"({int(x[1])}, {x[2]:.2g})" if x else " " for x in candidates_row])


def processed_list_to_latex_table(candidates_list):
    transposed_list = [[x[idx] if len(x) > idx else None for x in candidates_list] for idx in
                       range(len(max(candidates_list, key=len)))]
    return " \\\\ \\hline\n".join([latex_table_double_row(transposed_row) for transposed_row in transposed_list])


def relevance_to_colored_text(relevance_dict, input_instance):
    output_text = ''

    sentence_relevance_dict = relevance_dict
    sentence_value_list = list(sentence_relevance_dict.values())
    max_value = np.abs(np.array(sentence_value_list)).max()
    normalized_sentence_value_list = sentence_value_list / max_value

    for word, score in zip(input_instance.token_fields['sent']._tokens, normalized_sentence_value_list):
        red = 255 * min(1, 1 + score)
        green = 255 * (1 - abs(score))
        blue = 255 * min(1, 1 - score)
        output_text += '\colorbox[RGB]{' + str(int(red)) + ',' + str(int(green)) + ',' + str(int(blue)) + '}{\strut ' +\
                       word + '} '

    return output_text, max_value


def colored_text_to_table(relevance_dict, input_instance):
    table_string_start = ["\\begin{table*}[h]", "  \\centering", "  \\begin{tabular}{l|l|l}",
                          "    method&relevances&maximum value \\\ \hline"]
    table_string_end = ["  \\end{tabular}", "  \\caption{Example explanations for SST-2}",
                        "  \\label{tab:example_explanations}", "\\end{table*}"]

    for key, value_dict in relevance_dict.items():
        text, max_value = relevance_to_colored_text(value_dict, input_instance)
        table_string_start.append(f"    {key}&{text}&{'%.2g' % max_value}\\\\")

    return "\n".join(table_string_start + table_string_end)


def colored_text_to_table_cola(relevance_dict, input_instances, idc, method):
    table_string_start = ["\\begin{table*}[h]", "  \\centering", "  \\begin{tabular}{l|l|l}",
                          "    method&relevances&maximum value \\\ \hline"]
    table_string_end = ["  \\end{tabular}", "  \\caption{Example explanations for SST-2}",
                        "  \\label{tab:example_explanations}", "\\end{table*}"]

    for idx in idc:
        text, max_value = relevance_to_colored_text(relevance_dict[method][idx], input_instances[idx])
        table_string_start.append(f"    {method}&{text}&{'%.2g' % max_value}\\\\")

    return "\n".join(table_string_start + table_string_end)
