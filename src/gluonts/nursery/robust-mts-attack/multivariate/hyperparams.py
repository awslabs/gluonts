from typing import NamedTuple, List, Optional

# hybridize fails for now
hybridize = False


class Hyperparams(NamedTuple):
    num_eval_samples: int = 400
    epochs: int = 100
    num_batches_per_epoch: int = 100
    rank: int = 10
    num_cells: int = 40
    num_layers: int = 2
    conditioning_length: int = 100
    patience: int = 5
    cell_type: str = "lstm"
    dropout_rate: float = 0.01
    hybridize: bool = True
    learning_rate: float = 1e-3
    learning_rate_fullrank: float = 1e-5
    minimum_learning_rate: float = 1e-5
    batch_size: int = 16
    target_dim_sample: int = 2
    lags_seq: Optional[List[int]] = None
    scaling: bool = False


class FastHyperparams(NamedTuple):
    p = Hyperparams()
    epochs: int = 1
    num_batches_per_epoch: int = 1
    num_cells: int = 1
    num_layers: int = 1
    num_eval_samples: int = 1
    cell_type: str = "lstm"
    conditioning_length: int = 10
    batch_size: int = 16
    rank: int = 5

    target_dim_sample: int = p.target_dim_sample
    patience: int = p.patience
    hybridize: bool = hybridize
    learning_rate: float = p.learning_rate
    learning_rate_fullrank: float = p.learning_rate_fullrank
    minimum_learning_rate: float = p.minimum_learning_rate
    dropout_rate: float = p.dropout_rate
    lags_seq: Optional[List[int]] = p.lags_seq
    scaling: bool = p.scaling


if __name__ == '__main__':
    params = FastHyperparams()
    print(repr(params))
