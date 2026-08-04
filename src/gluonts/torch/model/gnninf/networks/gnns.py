# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# If you use this code in your work please cite:
# Multivariate Time Series Forecasting with Latent Graph Inference
# (https://arxiv.org/abs/2203.03423)


from torch import nn
import torch
from .gcl import GCL


class GNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        out_node_nf,
        in_edge_nf=0,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        residual=True,
        attention=False,
        del_edges=False,
        C=1.0,
        edge_nf_ratio=2,
        clamp=False,
    ):
        super(GNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.in_node_nf = in_node_nf
        self.C = C
        self.del_edges = del_edges
        self.clamp = clamp
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    del_edges=del_edges,
                    edge_nf_ratio=edge_nf_ratio,
                    clamp=self.clamp,
                ),
            )
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, retrieve_edges=False):
        inf_edges = []
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, att_val = self._modules["gcl_%d" % i](
                h, edges, edge_attr=edge_attr, C=self.C
            )
            if retrieve_edges:
                inf_edges.append(
                    {"edge_indexes": edges, "edge_values": att_val}
                )
        h = self.embedding_out(h)

        return h, inf_edges


class FC_GNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FC_GNN, self).__init__()
        in_edge_nf = 0
        self.gnn = GNN(*args, in_edge_nf=in_edge_nf, **kwargs)
        self.to(self.gnn.device)
        self.edges_dict = {}

    def forward(self, h, retrieve_edges=False):
        bs, num_sensor, _ = h.size()
        edges = self.get_edges(num_sensor, bs)
        h = h.view(bs * num_sensor, -1)
        return self.gnn(
            h, edges, edge_attr=None, retrieve_edges=retrieve_edges
        )

    def get_edges(self, n_nodes, bs):
        if self.gnn.del_edges:
            return None
        else:
            if n_nodes not in self.edges_dict:
                self.edges_dict[n_nodes] = {}
            if bs not in self.edges_dict[n_nodes]:
                edges, _ = cast_edges_batch(n_nodes, bs)
                self.edges_dict[n_nodes][bs] = edges
            [rows, cols] = self.edges_dict[n_nodes][bs]
            return [rows.to(self.gnn.device), cols.to(self.gnn.device)]


class AsynchronousGNN(GNN):
    def __init__(self, num_latent_nodes, different_mlps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_latent_nodes = num_latent_nodes
        self.different_mlps = different_mlps

    def forward(
        self,
        h,
        edges_a,
        edges_b,
        update_mask_a,
        update_mask_b,
        retrieve_edges=False,
    ):
        inf_edges = []
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            if i % 2 == 0:
                edges = edges_a
                update_mask = update_mask_a
                C = self.C
            else:
                edges = edges_b
                update_mask = update_mask_b
                C = 2.0 / (self.num_latent_nodes)
            if self.different_mlps is False:
                i = int(i / 2)
            h, att_val = self._modules["gcl_%d" % i](
                h, edges, update_mask=update_mask, C=C
            )
            if retrieve_edges:
                inf_edges.append(
                    {"edge_indexes": edges, "edge_values": att_val}
                )
        h = self.embedding_out(h)
        return h, inf_edges


class BP_asynchron_GNN(nn.Module):
    def __init__(
        self,
        num_latent_nodes,
        id_embedding_size,
        different_mlps,
        *args,
        **kwargs
    ):
        super(BP_asynchron_GNN, self).__init__()
        self.gnn = AsynchronousGNN(
            num_latent_nodes, different_mlps, *args, **kwargs
        )
        self.to(self.gnn.device)
        self.num_latent_nodes = num_latent_nodes
        self.edges_dict = {}
        self.id_latent = nn.Parameter(
            torch.randn(1, self.num_latent_nodes, id_embedding_size)
        )
        self.linear = nn.Linear(id_embedding_size, self.gnn.in_node_nf)

    def forward(self, h, retrieve_edges=False):
        bs, num_sensor, _ = h.size()

        # Get edges for the bipartite graph
        edges_a, edges_b = self.get_edges(
            num_sensor, self.num_latent_nodes, bs
        )

        # Get latent nodes embedding
        id_latent = self.id_latent.repeat(bs, 1, 1)
        id_latent = id_latent.view(bs * self.num_latent_nodes, -1)
        latent_emb = self.linear(id_latent).view(bs, self.num_latent_nodes, -1)

        # Concatenate latent nodes
        h = torch.cat([h, latent_emb], dim=1)
        update_mask_a, update_mask_b = self.get_masks(
            bs, num_sensor, self.num_latent_nodes
        )
        # Reshape to fit into the GNN
        h = h.view(bs * (num_sensor + self.num_latent_nodes), -1)

        h, inf_edges = self.gnn(
            h,
            edges_a,
            edges_b,
            update_mask_a,
            update_mask_b,
            retrieve_edges=retrieve_edges,
        )
        h = h.view(bs, num_sensor + self.num_latent_nodes, -1)

        # Remove the latent nodes again
        h = h[:, 0:num_sensor, :]
        return h, inf_edges

    def get_masks(self, bs, n_nodes, n_latent_nodes):
        mask_nodes = torch.zeros(bs, n_nodes, 1)
        mask_latent_nodes = torch.ones(bs, n_latent_nodes, 1)
        mask_a = torch.cat([mask_nodes, mask_latent_nodes], dim=1)
        mask_a = mask_a.view(bs * (n_nodes + n_latent_nodes), 1)
        mask_b = 1 - mask_a
        return (
            mask_a.to(self.gnn.device),
            mask_b.to(self.gnn.device),
        )

    def get_edges(self, n_nodes, num_latent_nodes, bs):
        key = (n_nodes, num_latent_nodes)
        if key not in self.edges_dict:
            self.edges_dict[key] = {}
        if bs not in self.edges_dict[key]:
            edges_a, edges_b = get_bip_edges_sets(n_nodes, num_latent_nodes)
            edges_a, _ = cast_edges_batch(
                n_nodes + num_latent_nodes, bs, edges_a
            )
            edges_b, _ = cast_edges_batch(
                n_nodes + num_latent_nodes, bs, edges_b
            )
            self.edges_dict[key][bs] = [edges_a, edges_b]

        [[rows_a, cols_a], [rows_b, cols_b]] = self.edges_dict[key][bs]
        return [
            [rows_a.to(self.gnn.device), cols_a.to(self.gnn.device)],
            [rows_b.to(self.gnn.device), cols_b.to(self.gnn.device)],
        ]


def get_fc_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    edges = [rows, cols]
    return edges


def get_bip_edges_sets(n_nodes_a, n_nodes_b):
    rows_a, cols_a = [], []
    rows_b, cols_b = [], []
    for i in range(n_nodes_a):
        for j in range(n_nodes_a, n_nodes_a + n_nodes_b):
            assert i != j
            rows_a.append(j)
            cols_a.append(i)
            rows_b.append(i)
            cols_b.append(j)
    edges = [[rows_a, cols_a], [rows_b, cols_b]]
    return edges


def cast_edges_batch(n_nodes, batch_size, edges=None):
    if edges is None:
        edges = get_fc_edges(
            n_nodes
        )  # If no edges are provided load fully connected edges
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr
