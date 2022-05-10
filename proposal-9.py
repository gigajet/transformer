from distutils.log import error
import torch
from torch import nn
from torch.nn import functional as F

from layer.FuzzyRule import MembershipFunctionLayer
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from typing import Optional
from fairseq.models import FairseqEncoder, FairseqDecoder, FairseqEncoderDecoderModel, register_model, register_model_architecture
from fairseq import utils
from mymodel.models.layer.PositionalEncodedEmbedding import PositionalEncodedEmbedding
from mymodel.models.layer.FuzzyRule import FuzzyRuleLayer
from mymodel.models.nnFairseqTransformer import NNTransformerDecoder

class Proposal9EncoderLayer(nn.TransformerEncoderLayer):
    r""""
    PROPOSAL 9:
    Fuzzy Layer is parallel to original Encoder, as another feature extractor

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Proposal9EncoderLayer, self).__init__(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation=activation,
            layer_norm_eps=layer_norm_eps, 
            batch_first=batch_first, 
            norm_first=norm_first,
            device=device, 
            dtype=dtype)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = F._get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + super()._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + super()._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + super()._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + super()._ff_block(x))
        return x


class Proposal9Encoder (FairseqEncoder):
    def __init__(self, max_src_len: int, dictionary, num_layer: int,
        dim_fuzzy: int,
        dim_model: int, dim_feedforward: int, num_head: int, dropout: float):
        super().__init__(dictionary)
        self.dictionary = dictionary
        self.src_pad_idx = dictionary.pad()

        self.embedding = PositionalEncodedEmbedding(max_src_len, dim_model, len(dictionary), dictionary.pad())
        encoder_layer = Proposal9EncoderLayer(d_model=dim_model,
            nhead=num_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True)
        self.nn_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
            num_layers=num_layer)
        self.fuzzy_membership = MembershipFunctionLayer(dim_model, dim_fuzzy)
        self.fuzzy_rule = FuzzyRuleLayer()

    def _fuzzy_block (self, x):
        x = self.fuzzy_rule(self.fuzzy_membership(x))
        return x

    """
    src_tokens: (batch, src_len)
    src_lengths: (batch)
    output: 
        context: (batch, src_len, dim_model)
    """
    def forward(self, src_tokens, src_lengths):
        """
        src (batch,src_len,E)
        mask (src_len,src_len)
        src_key_padding (batch,src_len)
        where S is the source sequence length, T is the target sequence length, N is the
        batch size, E is the feature number
        """
        src = self.embedding(src_tokens)
        mask = None
        src_key_padding = src_tokens.eq(self.src_pad_idx)

        x_fuzzy = self._fuzzy_block(src)
        x = self.nn_encoder(src, mask, src_key_padding)
        x = torch.cat((x,x_fuzzy), dim=-1)

        return {
            'context' : x,
            'src_tokens' : src_tokens,
            'src_lengths' : src_lengths,
            'src_key_padding' : src_key_padding,
            'src_pad_idx' : self.src_pad_idx
        }

    # Encoders are required to implement this method so that we can rearrange
    # the order of the batch elements during inference (e.g., beam search).
    # TODO: WTF IS THIS FUNCTION
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        # final_hidden = encoder_out['final_hidden']
        # return {
        #     'final_hidden': final_hidden.index_select(0, new_order),
        # }
        return {
            'context' : encoder_out['context'].index_select(0, new_order),
            'src_tokens' : encoder_out['src_tokens'].index_select(0, new_order),
            'src_key_padding' : encoder_out['src_key_padding'].index_select(0, new_order),
            'src_pad_idx' : self.src_pad_idx,
            'src_lengths' : encoder_out['src_lengths'].index_select(0, new_order)
        }

@register_model('proposal9')
class Proposal9Transformer(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add some new command-line arguments to configure dropout
        # and the dimensionality of the embeddings and hidden states.
        parser.add_argument(
            '--num-layer', type=int, metavar='N',
            help='number of layers in encoder (decoder)',
        )
        parser.add_argument(
            '--dim-fuzzy', type=int, metavar='N',
            help='dimensionality of the fuzzy rule layer',
        )
        parser.add_argument(
            '--dim-model', type=int, metavar='N',
            help='dimensionality of the embedding',
        )
        parser.add_argument(
            '--dim-feedforward', type=int, metavar='N',
            help='dimensionality of the feedforward layer inside encoder and decoder',
        )
        parser.add_argument(
            '--num-head', type=int, metavar='N',
            help='number of attention head.',
        )
        parser.add_argument(
            '--dropout', type=float, default=0.1,
            help='encoder and decoder dropout probability (default 0.1)',
        )
        parser.add_argument(
            '--max-src-len', type=int,
            help='Maximum number of tokens in source sequence'
        )
        parser.add_argument(
            '--max-tgt-len', type=int,
            help='Maximum number of tokens in target sequence'
        )

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.

        # Initialize our Encoder and Decoder.
        assert args.dim_fuzzy % args.num_head == 0, \
            "In proposal 9, num_head should divide dim_fuzzy"
        encoder = Proposal9Encoder(args.max_src_len,
            dictionary=task.source_dictionary,
            dim_fuzzy=args.dim_fuzzy,
            num_layer=args.num_layer,
            dim_model=args.dim_model,
            dim_feedforward= args.dim_feedforward,
            num_head=args.num_head,
            dropout=args.dropout)

        decoder = NNTransformerDecoder(args.max_tgt_len,
            dictionary=task.target_dictionary,
            num_layer=args.num_layer,
            dim_model=args.dim_model + args.dim_fuzzy,
            dim_feedforward= args.dim_feedforward,
            num_head=args.num_head,
            dropout=args.dropout
        )
        model = Proposal9Transformer(encoder, decoder)
        return model

@register_model_architecture('proposal9', 'proposal9_default')
def mytransformer_default(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.num_layer = getattr(args, 'num_layer', 6)
    args.dim_fuzzy = getattr(args, 'dim_fuzzy', 64)
    args.dim_model = getattr(args, 'dim_model', 448)
    args.dim_feedforward = getattr(args, 'dim_feedforward', 2048)
    args.num_head = getattr(args, 'num_head', 8)
    args.max_src_len = getattr(args, 'max_src_len', 4096)
    args.max_tgt_len = getattr(args, 'max_tgt_len', 4096)
