"""
A graph neural network model
to estimate the expected cost of a state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_add_pool, global_mean_pool

class PrepareEnvGCN(nn.Module):
    name = "PrepareEnvGCN"

    def __init__(self, args=None):
        super(PrepareEnvGCN, self).__init__()
        self.args = args
        self.conv1 = SAGEConv(13, 16)
        self.conv2 = SAGEConv(16, 8)
        self.conv3 = SAGEConv(8, 4)
        self.classifier = nn.Linear(4 * 2, 1)

    def forward(self, data, device):
        data = data.to(device)
        x, edge_index, batch_index = (
            data.x,
            data.edge_index,
            data.batch,
        )

        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = x = torch.cat(
            [global_mean_pool(x, batch_index), global_add_pool(x, batch_index)], dim=1
        )
        x = self.classifier(x)
        return x

    def loss(self, nn_out, data, device="cpu", writer=None, index=None):
        y = data.exc.to(device)
        op = nn_out[:, 0]
        loss = nn.L1Loss()
        loss_tot = loss(op, y)
        # Logging
        if writer is not None:
            writer.add_scalar("Loss/total_loss", loss_tot.item(), index)

        return loss_tot

    @classmethod
    def get_net_eval_fn(_, network_file, device):
        model = PrepareEnvGCN()
        model.load_state_dict(torch.load(network_file))
        model.eval()
        model.to(device)

        def prepare_net(graph):
            batch_idx = torch.tensor(
                [0 for i in range(graph["num_nodes"])], dtype=torch.int64
            )
            gcn_data = Data(
                x=graph["graph_nodes"],
                edge_index=graph["graph_edge_index"],
                batch=batch_idx,
            )
            with torch.no_grad():
                out = model(gcn_data, device)
                out = out[:, 0].detach().cpu().numpy()
                return out[0]

        return prepare_net