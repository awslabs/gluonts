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

import torch
from torch import nn
from . import gnns
from . import utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GNNInfNetwork(torch.nn.Module):
    '''
    Input and Output tensor will have the shape:
    (batch_size, num_nodes, seq_len, dimensions)
    :param in_channels: number of channels/dimensions in the input sequence (this value is usually set to 1)
    :param out_channels: number of channels/dimensions in the output sequence (this value is usually set to 1)
    :param input_length: input sequence length
    :param pred_length: output sequence length
    :param num_nodes: number of nodes/variables in the time series pannel
    :param nf: number of features in the hidden layers
    :param nf_enc: number of features in the hidden layers of the encoder
    :param attention: Whether using attention or not True/False
    :param del_edges: Set True to remove edges (NE-GNN baseline)
    :param use_unique_id: Use unique identifier for the nodes, we recommend setting it True
    :param enc_name: encoder network, it can be "mlpres" or "cnnres"
    :param dec_name: decoder network, it can only be "mlpres"
    :param agg_name: aggregation type 'gnn' for the FCGNN or 'bpgnn' for the BPGNN
    :param n_latent_nodes: number of axualiar/latent nodes in the BPGNN
    :param gnn_layers: number of layers in the gnn
    :param id_embedding_size: embedding size of the unique ids
    :param cnn_stride: strides in case of using "cnnres" as an encoder
    :param enc_layers: number of layers in the encoder in case of using "mlpres"
    :param dec_layers: number of layers in the decoder
    :param edge_nf_ratio: the number of features in edges is divided by nf/edge_nf_ratio
    :param residual: we recommend leaving it True
    :param clamp: whether clamping or not the attention values
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_length: int,
        pred_length: int,
        num_nodes: int,
        nf: int=64,
        nf_enc: int=64,
        attention: bool=True,
        del_edges: bool=False,
        use_unique_id: bool=True,
        enc_name: str='mlpres',
        dec_name: str='mlpres',
        agg_name: str='gnn',
        n_latent_nodes: int=4,
        gnn_layers: int=2,
        id_embedding_size: int=10,
        cnn_stride: int=6,
        enc_layers: int=2,
        dec_layers: int=1,
        edge_nf_ratio: int=2,
        residual: bool=True,
        clamp: bool=False
    ) -> None:

        super(GNNInfNetwork, self).__init__()
        self.pred_length = pred_length
        self.input_length = input_length
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.use_unique_id = use_unique_id
        self.id_embedding_size = id_embedding_size * use_unique_id
        self.enc_name = enc_name
        self.dec_name = dec_name
        self.agg_name = agg_name
        self.nf = nf
        self.gnn_layers = gnn_layers
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.edges_dict = {}
        self.nf_enc = nf_enc

        # Encoder
        if self.enc_name == 'mlpres':
            layers = [nn.Linear(in_channels * input_length, self.nf_enc)]
            for i in range(self.enc_layers):
                layers.append(utils.MLP(self.nf_enc, self.nf_enc, self.nf_enc, n_layers=2, residual=True))
            self.enc = nn.Sequential(*layers)
        elif self.enc_name == 'cnnres':
            self.enc = utils.CNNResidual(in_channels, self.nf_enc, input_length, nf=64, max_nf=self.nf_enc,
                                         stride=cnn_stride)
        elif self.enc_name == 'linear':
            self.enc = nn.Linear(in_channels * input_length, self.nf_enc)
        else:
            raise Exception('wrong encoder name %s' % self.enc_name)

        # Decoder
        if self.dec_name == 'mlpres':
            layers = []
            for i in range(self.dec_layers):
                layers.append(utils.MLP(nf, nf, nf, n_layers=2, residual=True))
            layers.append(nn.Linear(nf, out_channels * pred_length))
            self.dec = nn.Sequential(*layers)
        else:
            raise Exception('wrong decoder name %s' % self.dec_name)

        # Aggregation
        C = 2 / (max(1, self.num_nodes - 1))
        if self.agg_name == 'gnn':
            self.gnn = gnns.FC_GNN(self.nf_enc + self.id_embedding_size, nf, nf, n_layers=gnn_layers, residual=residual, device=device, act_fn=nn.SiLU(), attention=attention, del_edges=del_edges, C=C, edge_nf_ratio=edge_nf_ratio, clamp=clamp)
        elif self.agg_name == 'bpgnn':
            gnn_layers = gnn_layers * 2
            self.gnn = gnns.BP_asynchron_GNN(n_latent_nodes, id_embedding_size, True, self.nf_enc + self.id_embedding_size, nf, nf, n_layers=gnn_layers, residual=residual, device=device,
                                             act_fn=nn.SiLU(), attention=attention, del_edges=del_edges, C=C, edge_nf_ratio=edge_nf_ratio, clamp=clamp)
        else:
            raise Exception('Wrong agg model')

        self.unique_id = nn.Parameter(torch.randn(1, self.num_nodes, self.id_embedding_size))

    def forward(self, inputs: torch.Tensor, retrieve_edges: bool=False) -> torch.Tensor:
        '''
        :param inputs: shape (batch_size, num_nodes, seq_len, input_dim)
        :return: shape (batch_size, num_nodes, seq_len, out_channels)
        '''

        bs, num_sensor, seq_len, input_dim = inputs.size()

        #Encode
        h0 = self.encode(inputs)

        # Append a unique id to each node
        if self.use_unique_id:
            unique_id = self.unique_id.repeat(bs, 1, 1).view(bs * num_sensor, -1)
            h0 = torch.cat([h0, unique_id], dim=1)
        h0 = h0.view(bs, num_sensor, -1)

        # Run our aggregation model
        hL, inf_edges = self.gnn(h0, retrieve_edges=retrieve_edges)

        # Run decoder
        outputs = self.dec(hL)

        # Reshape it back to the original
        outputs = outputs.view(bs, num_sensor, self.pred_length, self.out_channels)

        if retrieve_edges:
            return outputs, inf_edges
        else:
            return outputs

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        # input shape (bs, num_sensor, seq_len, input_dim)
        bs, num_sensor, seq_len, input_dim = inputs.size()
        if self.enc_name == 'mlpres' or self.enc_name =='linear':
            inputs = inputs.reshape(bs*num_sensor, seq_len * input_dim)
        elif self.enc_name == 'cnnres':
            inputs = inputs.reshape(bs * num_sensor, seq_len, input_dim).transpose(1, 2)
            #  bs * num_sensor, input_dim, seq_len
        else:
            raise Exception('Wrong encodern ame %s')
        h0 = self.enc(inputs)

        return h0  # return shape (bs*num_sensor, nf)

