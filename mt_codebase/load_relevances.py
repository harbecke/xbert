import os
import dill


def experiment_load_relevances(experiment_dir: str,
                               relevance_filename: str = "relevances.pkl"):

    experiment_relevances = {}
    for subfolder in os.listdir(experiment_dir):
        relevance_file = os.path.join(experiment_dir, subfolder, relevance_filename)
        with open(relevance_file, "rb") as f:
            relevances = dill.load(f)
            experiment_relevances[subfolder] = relevances

    return experiment_relevances
