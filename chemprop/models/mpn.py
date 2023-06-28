from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import SuperGATConv,GatedGraphConv,GATv2Conv,GATConv,TransformerConv


from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function
import math
import torch.nn.functional as F


from typing import Optional
from torch import Tensor
from torch_scatter import scatter_add

from torch_geometric.utils import softmax


class Set2Set(torch.nn.Module):
    r"""The global pooling operator based on iterative content-based attention
    from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
            :obj:`num_layers=2` would mean stacking two LSTMs together to form
            a stacked LSTM, with the second LSTM taking in outputs of the first
            LSTM and computing the final results. (default: :obj:`1`)

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          batch vector :math:`(|\mathcal{V}|)` *(optional)*
        - **output:** graph features :math:`(|\mathcal{G}|, 2 * F)` where
          :math:`|\mathcal{G}|` denotes the number of graphs in the batch
    """
    def __init__(self, in_channels: int, processing_steps: int,
                 num_layers: int = 1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels,
                                  num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x: Tensor, batch: Optional[Tensor] = None,
                size: Optional[int] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): The input node features.
            batch (LongTensor, optional): A vector that maps each node to its
                respective graph identifier. (default: :obj:`None`)
            size (int, optional): The number of graphs in the batch.
                (default: :obj:`None`)
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.int64)

        size = int(batch.max()) + 1 if size is None else size

        h = (x.new_zeros((self.num_layers, size, self.in_channels)),
             x.new_zeros((self.num_layers, size, self.in_channels)))
        q_star = x.new_zeros(size, self.out_channels)

        for _ in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(size, self.in_channels)
            e = (x * q.index_select(0, batch)).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=size)
            r = scatter_add(a * x, batch, dim=0, dim_size=size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')



class MPNEncoder(nn.Module):
    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        # self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)
        # Input
        input_dim = self.atom_fdim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.bond_fdim
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)

        w_h_input_size_bond = self.hidden_size


        self.attention = TransformerConv(self.hidden_size,self.hidden_size,edge_dim=self.hidden_size,heads=8,concat=False)


        self.set2set = Set2Set(self.hidden_size,1)



        for depth in range(self.depth-1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.hidden_size*2, self.hidden_size)

        # self.gru = BatchGRU(self.hidden_size)


        self.lr = nn.Linear(self.hidden_size*8, self.hidden_size, bias=self.bias)


    def forward(self, mol_graph: BatchMolGraph, features_batch=None) -> torch.FloatTensor:

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = mol_graph.get_components()
        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb,bonds = (f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda(),bonds.cuda())
        # Input

        input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_sizez
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()

        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        input_bond = self.act_func(input_bond)
        message_bond = input_bond.clone()

        # Message passing
        for depth in range(self.depth-1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message

            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden
            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))

        atom_hiddens = self.attention(message_atom,bonds,message_bond)
        # atom_hiddens = self.lr(atom_hiddens)
        # atom_hiddens = self.dropout_layer(self.act_func(atom_hiddens))

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            # cur_hiddens = cur_hiddens.mean(0)
            cur_hiddens = self.set2set(cur_hiddens)[0]
            cur_hiddens = self.W_o(cur_hiddens)
            mol_vecs.append(cur_hiddens)
        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs  # B x H



class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # print(message.shape)
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            # print(a_start, a_size, cur_message.shape, cur_hidden.shape)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))

            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_atom_len-cur_message.shape[0]))(cur_message)
            # print(cur_message.shape)
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)       # (batch, MAX_atom_len, hidden)
        hidden_lst = torch.cat(hidden_lst, 1)         # (1, batch, hidden)
        hidden_lst = hidden_lst.repeat(2, 1, 1)       # (2, batch, hidden)
        # print(message_lst.shape, hidden_lst.shape)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)     # message = (batch, MAX_atom_len, 2 * hidden)
        # print(cur_message.shape, cur_hidden.shape)
        # print()

        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2*self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
        # print(cur_message_unpadding.shape, message.narrow(0, 0, 1).shape)

        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
                             cur_message_unpadding], 0)
        # print(message.shape)
        return message


class MPN(nn.Module):
    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + args.atom_messages * self.atom_fdim  # * 2
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                smiles_batch: Union[List[str], BatchMolGraph],
                crystals_batch,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        if not self.graph_input:  # if features only, batch won't even be used
            # batch = mol2graph(smiles_batch, self.args)
            batch = mol2graph(smiles_batch, crystals_batch, self.args)
        else:
            batch = smiles_batch

        return self.encoder.forward(batch, features_batch)

