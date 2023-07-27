import os.path as osp
import sys
import warnings
from itertools import repeat

from collections import defaultdict

import pandas as pd


import torch

from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import coalesce, index_to_mask, remove_self_loops


def read_planetoid_data(df, labels):

    x, paper2id = load_embedding(df)

    graph = build_graph(df, paper2id)

    edge_index = edge_index_from_dict(graph, num_nodes=df.shape[0])

    y = load_y(df, labels)

    train_mask, val_mask, test_mask = load_mask(df)

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return data


def load_mask(df):
    train_data = pd.read_json('data/MeSH/train.json', lines=True)
    val_data = pd.read_json('data/MeSH/dev.json', lines=True)
    test_data = pd.read_json('data/MeSH/test.json', lines=True)

    total_paper = df['paper'].to_list()

    train_mask = [False]*df.shape[0]

    for paper_id in train_data['paper'].to_list():
        if paper_id in total_paper:
            train_mask[total_paper.index(paper_id)] = True

    val_mask = [False] * df.shape[0]

    for paper_id in val_data['paper'].to_list():
        if paper_id in total_paper:
            val_mask[total_paper.index(paper_id)] = True

    test_mask = [False] * df.shape[0]

    for paper_id in test_data['paper'].to_list():
        if paper_id in total_paper:
            test_mask[total_paper.index(paper_id)] = True

    return torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(test_mask)


def load_y(df, labels):
    # with open('data/MeSH/id2label.txt','r') as f:
    #     y_list = [line.split('	')[0] for line in f.readlines()]
    res = []
    for i in range(df.shape[0]):
        y = [0]*len(labels)

        for label in df.iloc[i]['label']:
            y[labels.index(label)] = 1
        res.append(y)

    return torch.tensor(res, dtype=torch.float32)


def load_embedding(df):

    paper2id = dict()

    x = torch.squeeze(torch.tensor(df["embedding"]), 1)

    for i in range(df.shape[0]):

        paper2id[str(df.iloc[i]["paper"])] = i

    return x, paper2id


def build_graph(df, paper2id):

    graph = defaultdict(list)

    for i in range(df.shape[0]):

        for j in range(len(df.iloc[i]["reference"])):

            if df.iloc[i]["reference"][j] in paper2id.keys():

                graph[paper2id[str(df.iloc[i]["paper"])]].append(paper2id[df.iloc[i]["reference"][j]])

    return graph


def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)

    # NOTE: There are some duplicated edges and self loops in the datasets.
    #       Other implementations do not remove them!
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return edge_index
