class TransformerConfig:
    def __init__(
        self,
        model_dim: int = 64,  # dimension of the embedded input
        # inner dimension of the feedforward network (dimension of the hidden layer) is multiple of the input dimension
        inner_ff_dim_scale: int = 4,
        pre_seq: str = 'dn',
        post_seq: str = 'drn',
        dropout: float = 0.1,
        act_type: str = 'relu',
        num_heads: int = 8,
    ) -> None:

        self.model_dim = model_dim

        # Encoder
        # pre self-attention block (dropout/residual connection/normalization)
        self.enc_pre_self_att_sequence = pre_seq
        # pre self-attention block dropout rate
        self.enc_pre_self_att_dropout = dropout
        # self-attention input dimension (data dimension -> input dimension)
        self.enc_self_att_dim_in = model_dim
        # number of self-attention heads
        self.enc_self_att_heads = num_heads
        # self-attention dropout rate
        self.enc_self_att_dropout = dropout
        # post self-attention block (dropout/residual connection/normalization)
        self.enc_post_self_att_sequence = post_seq
        # post self-attention block dropout rate
        self.enc_post_self_att_dropout = dropout
        # inner dimension (hidden layer) of the feed forward network
        self.enc_ff_inn_dim = model_dim * inner_ff_dim_scale
        # output dimension of the feed forward network
        self.enc_ff_out_dim = model_dim
        # activation function of the feed forward network
        self.enc_ff_act_type = act_type
        # dropout rate of the feed forward network
        self.enc_ff_dropout = dropout
        # post feed forward block (dropout/residual connection/normalization)
        self.enc_post_ff_sequence = post_seq
        # post feed forward dropout rate
        self.enc_post_ff_dropout = dropout

        # Decoder
        # pre self-attention block (dropout/residual connection/normalization)
        self.dec_pre_self_att_sequence = pre_seq
        # pre self-attention block dropout rate
        self.dec_pre_self_att_dropout = dropout
        # self-attention input dimension (data dimension -> input dimension)
        self.dec_self_att_dim_in = model_dim
        # number of self-attention heads
        self.dec_self_att_heads = num_heads
        # self-attention dropout rate
        self.dec_self_att_dropout = dropout
        # post self-attention block (dropout/residual connection/normalization)
        self.dec_post_self_att_sequence = post_seq
        # post self-attention block dropout rate
        self.dec_post_self_att_dropout = dropout
        # input dimension of the attention layer
        self.dec_att_dim_in = model_dim
        # number of attention heads
        self.dec_att_heads = num_heads
        # attention dropout rate
        self.dec_att_dropout = dropout

        self.dec_post_att_sequence = post_seq
        self.dec_post_att_dropout = dropout
        self.dec_ff_inn_dim = model_dim * inner_ff_dim_scale
        self.dec_ff_out_dim = model_dim
        self.dec_ff_act_type = act_type
        self.dec_ff_dropout = dropout
        self.dec_post_ff_sequence = post_seq
        self.dec_post_ff_dropout = dropout

    def __repr__(self):
        return "Config[%s]" % ", ".join(
            "%s=%s" % (str(k), str(v))
            for k, v in sorted(self.__dict__.items())
        )
