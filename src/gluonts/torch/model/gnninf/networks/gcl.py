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
# Multivariate Time Series Forecasting with Latent Graph Inference (https://arxiv.org/abs/2203.03423)


from torch import nn
import torch
from typing import Tuple, List

class GCL(nn.Module):
    def __init__(self,
        input_nf: int,
        output_nf: int,
        hidden_nf: int,
        edges_in_d: int=0,
        act_fn: nn.Module=nn.SiLU(),
        residual: bool=True,
        attention: bool=False,
        del_edges: bool=False,
        edge_nf_ratio: int=2,
        clamp=False
    ) -> None:
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.del_edges = del_edges
        self.clamp = clamp
        edges_nf = hidden_nf // edge_nf_ratio * (1-del_edges)
        out_edges_nf = edges_nf

        if not del_edges:
            self.edge_mlp = nn.Sequential(
                nn.Linear(input_edge + edges_in_d, edges_nf),
                act_fn,
                nn.Linear(edges_nf, out_edges_nf),
                act_fn)

            if self.attention:
                self.att_mlp = nn.Sequential(
                    nn.Linear(out_edges_nf, 1),
                    nn.Sigmoid())

        self.node_mlp = nn.Sequential(
            nn.Linear(out_edges_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if not del_edges:
            self.node_linear = nn.Linear(out_edges_nf, output_nf)

    def edge_model(self, source: torch.Tensor, target: torch.Tensor, edge_attr: torch.Tensor):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            if self.clamp:
                att_val = torch.clamp(att_val, min=0.1, max=0.9)
            out = out * att_val
        else:
            att_val = None
        return out, att_val

    def node_model(self, x: torch.Tensor, edge_index: List[torch.Tensor], edge_attr: torch.Tensor,
                   node_attr: torch.Tensor, update_mask: torch.Tensor, C: float) -> Tuple:
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0)) * C

        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)

        # This mask is used in the bipartite graph
        if update_mask is not None:
            out = out * update_mask

        if self.residual:
            out = x + out
        else:
            raise Exception('Warning: BPGNN has only been coded for the residual case')
        return out, agg

    def node_model_noedges(self, x: torch.Tensor, node_attr: torch.Tensor) -> torch.Tensor:
        if node_attr is not None:
            agg = torch.cat([x, node_attr], dim=1)
        else:
            agg = x
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out

    def forward(self, h: torch.Tensor, edge_index:torch.Tensor, edge_attr: torch.Tensor=None,
                node_attr: torch.Tensor=None, update_mask: torch.Tensor=None, C: float=1.0) -> Tuple:
        if self.del_edges:
            h = self.node_model_noedges(h, node_attr)
            att_val = None
        else:
            row, col = edge_index
            edge_feat, att_val = self.edge_model(h[row], h[col], edge_attr)
            h, _ = self.node_model(h, edge_index, edge_feat, node_attr, update_mask, C)

        return h, att_val


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result
