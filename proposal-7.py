import torch
from torch import nn
from torch.nn import functional as F

from layer.FuzzyRule import MembershipFunctionLayer
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from typing import Optional
from fairseq.models import FairseqEncoder, FairseqDecoder, FairseqEncoderDecoderModel, register_model, register_model_architecture
from fairseq import utils
from layer.PositionalEncodedEmbedding import PositionalEncodedEmbedding
from layer.FuzzyRule import FuzzyRuleLayer
from nnFairseqTransformer import NNTransformerDecoder, NNTransformerEncoder
from layer.PositionalEncoding import PositionalEncoding

"""
PROPOSAL 7
FuzzyLayer is inserted after Embedding and PositionalEncoding, before Encoder
"""

class Proposal7Encoder (FairseqEncoder):
    def __init__(self, max_src_len: int, dictionary, num_layer: int,
        dim_model: int, dim_feedforward: int, num_head: int, dropout: float):
        super().__init__(dictionary)
        self.dictionary = dictionary
        self.src_pad_idx = dictionary.pad()

        self.embedding = PositionalEncodedEmbedding(max_src_len, dim_model, len(dictionary), dictionary.pad())
        self.membership_layer = MembershipFunctionLayer(dim_model, dim_model)
        self.fuzzyrule_layer = FuzzyRuleLayer()

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
            nhead=num_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True)
        self.nn_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
            num_layers=num_layer)

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
        src = self.fuzzyrule_layer(self.membership_layer(src))

        mask = None
        src_key_padding = src_tokens.eq(self.src_pad_idx)
        x = self.nn_encoder(src, mask, src_key_padding)

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

@register_model('proposal7')
class Proposal7Transformer(FairseqEncoderDecoderModel):
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

        assert args.dim_model == args.dim_fuzzy, \
            "IN PROPOSAL 1-8, assert(dim_model == dim_fuzzy)"
        # Initialize our Encoder and Decoder.
        encoder = Proposal7Encoder(args.max_src_len,
            dictionary=task.source_dictionary,
            num_layer=args.num_layer,
            dim_model=args.dim_model,
            dim_feedforward= args.dim_feedforward,
            num_head=args.num_head,
            dropout=args.dropout)
        decoder = NNTransformerDecoder(args.max_tgt_len,
            dictionary=task.target_dictionary,
            num_layer=args.num_layer,
            dim_model=args.dim_model,
            dim_feedforward= args.dim_feedforward,
            num_head=args.num_head,
            dropout=args.dropout
        )
        model = Proposal7Transformer(encoder, decoder)
        return model

@register_model_architecture('proposal7', 'proposal7_default')
def mytransformer_default(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.num_layer = getattr(args, 'num_layer', 6)
    args.dim_model = getattr(args, 'dim_model', 128)
    args.dim_fuzzy = getattr(args, 'dim_fuzzy', args.dim_model) # UNUSED
    args.dim_feedforward = getattr(args, 'dim_feedforward', 2048)
    args.num_head = getattr(args, 'num_head', 8)
    args.max_src_len = getattr(args, 'max_src_len', 4096)
    args.max_tgt_len = getattr(args, 'max_tgt_len', 4096)
