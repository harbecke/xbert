import dill
import csv
from collections import defaultdict


with open("../results/cola/resampling/candidate_instances.pkl", "rb") as in_f:
    candidate_instances, candidate_results = dill.load(in_f)


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


def read_dataset_and_append_label(data1, data2, output):
    with open(data1, 'r') as csv_file_1:
        spamreader = csv.reader(csv_file_1, delimiter='\t')
        next(spamreader)
        for idx1, line in enumerate(spamreader):
            output[idx1].append(line[1])
            
    with open(data2, 'r') as csv_file_2:
        spamreader = csv.reader(csv_file_2, delimiter='\t')
        for idx2, line in enumerate(spamreader):
            instances_dict[idx1+idx2+1].append(line[1])
    
    return


# In[8]:


instances_dict, candidates_dict = create_candidate_dicts(candidate_instances, candidate_results)


# In[9]:


read_dataset_and_append_label("../results/cola/in_domain_dev.tsv",
                              '../results/cola/out_of_domain_dev.tsv',
                              instances_dict)


# In[10]:


instances_dict[0]


# In[11]:


candidates_dict[0][9][1]

