import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd


class FavaDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, delimiter="\t", usecols=[1, 3])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx, 1], self.df.iloc[idx, 0]


def run(dataset_file, batch_size, device):
    model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
    dataset = FavaDataset(dataset_file)
    params = {'batch_size': batch_size, 'shuffle': False}
    data_generator = DataLoader(dataset, **params)

    correct = 0
    for inputs, labels in data_generator:
        inputs = tokenizer(inputs, return_tensors="pt", padding=True)
        inputs, labels = inputs.to(device), labels.to(device)
        output = torch.argmax(model(**inputs)[0], 1)
        correct += int(sum(output==labels))

    return correct/len(dataset)


if __name__ == "__main__":
    result = run(dataset_file="data/fava/dev.tsv", batch_size=32, device="cuda:0")
    print(result)
