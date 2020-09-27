# OLM: Occlusion with Language Modeling

This is the repo for my Master's Thesis "Explaining Natural Language Processing Classifiers with Occlusion and Language Modeling".
It is build on a clone from a previous version of the paper repository https://github.com/DFKI-NLP/OLM.
Code that I wrote exclusively for the thesis can be found in the mt_codebase and mt_notebooks folders.
mt_codebase contains functions that I used in the notebooks and the evaluate_fava script that evaluates a model on FAVA.
mt_notebook contains notebooks with cached results.
All results mentioned in the paper are present.

## Installation

Install pytorch from https://pytorch.org/get-started/locally/

Then clone the repository to your machine and install the requirements with the following command:

```bash
pip install -r requirements.txt
```

## (if needed) Rerun Experiments

Unpack the results.zip from the root directory to not reproduce the explanation results from the paper experiments (takes hours) which are needed for some scripts.

Run mt_codebase/evaluate_fava and the notebooks in mt_notebooks.
