import torch
from torch import nn
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from typing import Optional
from fairseq.models import FairseqEncoder, FairseqDecoder, FairseqEncoderDecoderModel, register_model, register_model_architecture
from fairseq import utils
from mymodel.models.layer.TransformerDecoder import TransformerDecoder
from mymodel.models.layer.TransformerEncoder import TransformerEncoder
from mymodel.models.layer.PositionalEncodedEmbedding import PositionalEncodedEmbedding

class MyTransformerEncoder (FairseqEncoder):
    def __init__(self, args, dictionary, num_layer: int,
        dim_model: int, dim_feedforward: int, num_head: int, dropout: float):
        super().__init__(dictionary)
        self.dictionary = dictionary
        self.args = args
        self.embedding = PositionalEncodedEmbedding(10000, dim_model, len(dictionary), dictionary.pad())
        self.encoder = TransformerEncoder(num_layer, dim_model, dim_feedforward, num_head, dropout)
        self.src_pad_idx = dictionary.pad()

    """
    src_tokens: (batch, src_len)
    src_lengths: (batch)
    output: 
        context: (batch, src_len, dim_model)
    """
    def forward(self, src_tokens, src_lengths):

        if self.args.left_pad_source:
            # Convert left-padding to right-padding.
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                padding_idx=self.dictionary.pad(),
                left_to_right=True
            )

        x = self.embedding(src_tokens)
        x = self.encoder(x)
        # Return the Encoder's output. This can be any object and will be
        # passed directly to the Decoder.
        print('enc_forward',src_tokens, src_lengths)
        print('context',x.shape)

        return {
            'context' : x,
            'src_tokens' : src_tokens,
            'src_lengths' : src_lengths,
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
            'src_pad_idx' : self.src_pad_idx,
            'src_lengths' : encoder_out['src_lengths'].index_select(0, new_order)
        }

class MyTransformerDecoder (FairseqDecoder):

    def __init__(
        self, dictionary, num_layer: int,
        dim_model: int, dim_feedforward: int, num_head: int, dropout: float
    ):
        super().__init__(dictionary)
        self.embedding = PositionalEncodedEmbedding(10000, dim_model, len(dictionary), dictionary.pad())
        self.decoder = TransformerDecoder(num_layer, dim_model, dim_feedforward, num_head, dropout)
        # Define the output projection.
        self.output_projection = nn.Linear(dim_model, len(dictionary))
        self.dropout = nn.Dropout(dropout)

    # During training Decoders are expected to take the entire target sequence
    # (shifted right by one position) and produce logits over the vocabulary.
    # The *prev_output_tokens* tensor begins with the end-of-sentence symbol,
    # ``dictionary.eos()``, followed by the target sequence.
    """
        prev_output_tokens: (batch, tgt_len)
        encoder_out: (batch, src_len, dim_model)
        output_mask: (batch, tgt_len, tgt_len)
        context_mask: (batch, tgt_len, src_len)

        output: (batch, tgt_len, vocab)
    """
    def forward(self, prev_output_tokens, encoder_out):
        # Extract the final hidden state from the Encoder.
        context = encoder_out['context']
        output_mask = None
        context_mask = self.make_pad_mask(prev_output_tokens, encoder_out['src_tokens'], encoder_out['src_pad_idx'])

        # Embed the target sequence, which has been shifted right by one
        # position and now starts with the end-of-sentence symbol.
        x = self.embedding(prev_output_tokens) # (bsz, tgt_len, dim_model)

        x = self.decoder(x, context, output_mask, context_mask) # (bsz, tgt_len, dim_model)
        x = self.output_projection(x)
        x = self.dropout(x)
        return x, None

    """
     ____m______
     |
    n|
     |
    row: (batch, n)
    col: (batch, m)
    pad_idx: int?
    output: (batch,n,m) of Boolean where a[b,i,j] True iff col[b,j]=source_pad_idx
        or None if pad_idx is None
    """
    @staticmethod
    def make_pad_mask (row, col, source_pad_idx: Optional[int]=None):
        if source_pad_idx is None:
            return None
        n, m = row.size(-1), col.size(-1)
        masked = col.eq(source_pad_idx).unsqueeze(-2).repeat_interleave(n,-2) # # (batch,n,m)
        return masked.unsqueeze(-3)

@register_model('mytransformer')
class MyTransformer(FairseqEncoderDecoderModel):
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

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.

        # Initialize our Encoder and Decoder.
        encoder = MyTransformerEncoder(args,
            dictionary=task.source_dictionary,
            num_layer=args.num_layer,
            dim_model=args.dim_model,
            dim_feedforward= args.dim_feedforward,
            num_head=args.num_head,
            dropout=args.dropout)
        decoder = MyTransformerDecoder(
            dictionary=task.target_dictionary,
            num_layer=args.num_layer,
            dim_model=args.dim_model,
            dim_feedforward= args.dim_feedforward,
            num_head=args.num_head,
            dropout=args.dropout
        )
        model = MyTransformer(encoder, decoder)

        # Print the model architecture.
        print(model)

        return model

@register_model_architecture('mytransformer', 'mytransformer_default')
def mytransformer_default(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.num_layer = getattr(args, 'num_layer', 6)
    args.dim_model = getattr(args, 'dim_model', 512)
    args.dim_feedforward = getattr(args, 'dim_feedforward', 2048)
    args.num_head = getattr(args, 'num_head', 8)
