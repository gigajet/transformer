from fairseq.models import FairseqEncoder, FairseqDecoder, FairseqEncoderDecoderModel

from layer.TransformerEncoder import TransformerEncoder
from layer.PositionalEncodedEmbedding import PositionalEncodedEmbedding

class FairseqTransformerEncoder (FairseqEncoder):
    def __init__(self, dictionary, num_layer: int,
        dim_model: int, dim_feedforward: int, num_head: int, dropout: float):
        super().__init__(dictionary)
        self.embedding = PositionalEncodedEmbedding()
        self.encoder = TransformerEncoder(num_layer, dim_model, dim_feedforward, num_head, dropout)

    def forward(self, src_tokens, src_lengths):
        


class MyFairseqTransformer(FairseqEncoderDecoderModel):
    
    pass