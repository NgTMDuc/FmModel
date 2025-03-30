import pandas as  pd 
import numpy as np
import os
import json 
from datasets import load_dataset


def preprocess_function(examples):
    return {
        "input": [f"Questions: {q} \n{opts}" for q, opts in zip(examples["question"], examples["choices"])],
        "output": examples["answer"]
    }
def dataset_loader(mode):
    if mode == "train":
        dataset = load_dataset("csv", data_files = "b6_train_data.csv")["train"]
        dataset = dataset.map(preprocess_function, batched = True, remove_columns=dataset.column_names)
    
    return dataset

if __name__ == "__main__":
    print(dataset_loader("train")[0])