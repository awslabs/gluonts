import torch

from gluonts.torch.model.logsparse_transformer.module import DecoderTransformer

batch_size = 4
win_len = 20
input_dim = 3

model = DecoderTransformer(
    input_dim=input_dim,
    n_head=4,
    seq_num=5,
    layer=2,
    n_embd=3,
    win_len=win_len,
    scale_att=True,
    q_len=4,
    embd_pdrop=0.1,
    sparse=True,
    sub_len=2,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
)

# sanity checks

ids = torch.ones(size=(batch_size,), dtype=int)
x = torch.ones(size=(batch_size, win_len, input_dim))

mu, sigma = model(ids, x)

print(mu.shape)  # torch.Size([4, 20, 1])
print(sigma.shape)  # torch.Size([4, 20, 1])

# set up data loader for training

# simple training loop

# set up data loader for inference

# make predictions
