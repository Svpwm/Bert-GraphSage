import os.path
import pickle

import torch
import torch.nn as nn
import math
import json
import copy
import numpy as np
import pandas as pd


def extract_pubmed_data():
    # Load the PubMed dataset
    data = pd.read_json('data/MeSH/MeSH.json', lines=True)

    with open("data/MeSH/vocabulary.txt") as f:
        id2text = {item.strip("\n").split("	")[0]: item.strip("\n").split("	")[1] for item in f.readlines()}
        text2id = {item.strip("\n").split("	")[1]: item.strip("\n").split("	")[0] for item in f.readlines()}

    pooler_outputs = []
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', TruncationStrategy="only_first")
    model = BertModel.from_pretrained("bert-base-uncased")
    for text in data["text"]:
        text = " ".join([id2text[item] for item in text.split(" ")])
        encoded_input = tokenizer(text, return_tensors='pt')
        # if encoded_input.input_ids.shape[1]>512:
        encoded_input = {"input_ids": encoded_input.input_ids[..., :512],
                         "token_type_ids": encoded_input.token_type_ids[..., :512],
                         "attention_mask": encoded_input.attention_mask[..., :512]}

        output = model(**encoded_input)
        pooler_output = output.pooler_output.detach().numpy()
        pooler_outputs.append(pooler_output)
    data.insert(loc=0, column='embedding', value=pooler_outputs)
    with open('data/embedding', 'wb') as f:
        pickle.dump(data, f)
    return data


if __name__ == '__main__':

    if os.path.exists('data/embedding'):
        with open('data/embedding', 'rb') as f:
            data = pickle.load(f)
    else:
        data = extract_pubmed_data()
