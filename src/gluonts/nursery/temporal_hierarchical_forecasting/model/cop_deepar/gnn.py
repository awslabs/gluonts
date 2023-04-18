import mxnet as mx

from gluonts.mx import Tensor


class GNN(mx.gluon.HybridBlock):
    def __init__(
        self,
        units: int,
        num_layers: int,
        adj_matrix: Tensor,
        use_mlp: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units
        self.num_layers = num_layers
        self.adj_matrix = adj_matrix
        self.use_mlp = use_mlp

        if self.use_mlp:
            with self.name_scope():
                self.gnn_layer = mx.gluon.nn.Dense(
                    units=self.units, flatten=False
                )

    def hybrid_forward(self, F, x, *args, **kwargs):
        # Do message passing for `num_layers` times with learnable weights.
        for _ in range(self.num_layers):
            if self.use_mlp:
                x = x + self.gnn_layer(x)
                x = F.dot(x.swapaxes(-1, -2), self.adj_matrix).swapaxes(-1, -2)
                x = F.relu(x)
            else:
                x = F.dot(x.swapaxes(-1, -2), self.adj_matrix).swapaxes(-1, -2)

        return x
