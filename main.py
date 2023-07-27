import os.path as osp
import os
import pickle

from torch.nn import Linear
from torch_geometric.graphgym import SAGEConv
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from bert import extract_pubmed_data

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from torch_geometric.loader import NeighborLoader

from io_planetoid import read_planetoid_data


if os.path.exists('data/embedding'):
    with open('data/embedding', 'rb') as f:
        df = pickle.load(f)
else:
    df = extract_pubmed_data()

labels = []
for label_list in df["label"].to_list():
    labels.extend(label_list)

labels = list(set(labels))

data = read_planetoid_data(df, labels)

train_loader = NeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=8,
    input_nodes=data.train_mask,
)

val_loader = NeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=8,
    input_nodes=data.val_mask,
)

test_loader = NeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=8,
    input_nodes=data.test_mask,
)


class Net(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.linear = Linear(out_channels, len(labels))

    def forward(self, x, edge_index):

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        x = self.linear(x)
        x = torch.sigmoid(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = Net(in_channels=768, hidden_channels=1024, out_channels=1024).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
x, edge_index = data.x.to(device), data.edge_index.to(device)


def train():
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        loss = F.binary_cross_entropy(y_hat, y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        # total_correct += int((torch.ones(y_hat.shape, dtype=torch.float32)*y_hat.ge(0.5) == y).sum())
        total_correct += sum(row.all().int().item() for row in (y_hat.ge(0.5) == y))
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)

    pbar.close()

    return total_loss / total_examples, total_correct/total_examples


@torch.no_grad()
def val():
    model.eval()

    total_loss = total_correct = total_examples = 0

    for batch in val_loader:
        batch = batch.to(device)
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]

        total_loss += float(loss) * batch.batch_size
        total_correct += sum(row.all().int().item() for row in (y_hat.ge(0.5) == y))
        total_examples += batch.batch_size

    return total_loss / total_examples, total_correct/total_examples


@torch.no_grad()
def test():
    model.eval()

    total_loss = total_correct = total_examples = 0

    for batch in test_loader:
        batch = batch.to(device)
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]

        total_loss += float(loss) * batch.batch_size
        total_correct += sum(row.all().int().item() for row in (y_hat.ge(0.5) == y))
        total_examples += batch.batch_size

    return total_loss / total_examples, total_correct/total_examples


for epoch in range(10):
    loss, acc = train()
    val_loss, val_acc = val()
    test_loss, test_acc = test()
    print(f'\nEpoch: {epoch:03d}, Loss: {loss:.4f}, ACC: {acc:.4f}, '
          f'val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, '
          f'test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}')

    torch.save(model, f'model/net{epoch}.pt')
