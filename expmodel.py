import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#TODO: check if need to modify for the type of data we have
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class ExpModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        #vertices embedding
        self.embed_vertices = nn.Linear(args.vertices_dim, args.feat_dim)
        #emotion embedding
        self.embed_emotion = nn.Linear(args.emotion_dim, args.feat_dim)
        #positional encoding
        self.pos_enc = PositionalEncoding(args.feat_dim, args.dropout)
        #layernorm for the transformer decoder
        layer_norm = nn.LayerNorm(normalized_shape = ... )
        #transformer decoder
        dec_layer = nn.TransformerDecoderLayer(d_model = args.feat_dim, nhead = args.nhead, dim_feedforward = 2*args.feat_dim, batch_first = True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers = args.nlayer, norm = layer_norm)
        #TODO:define encoder for the emotions and determine if there's other to be created


    def forward(self, emotion, vertices, length):
        
        #embed the starting vertices
        x = self.embed_vertices(vertices)

        #embed emotion
        embedded_emotion = self.embed_emotion(emotion)

        #TODO: encoder for emotion


        #TODO: decoder 

        return x



    def predict(self, emotion, vertices, length):
        pass