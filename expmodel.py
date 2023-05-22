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
        dec_layer = nn.TransformerDecoderLayer(d_model = args.feat_dim, nhead = args.nhead_dec, dim_feedforward = 2*args.feat_dim, batch_first = True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers = args.nlayer_dec, norm = layer_norm)
        #define encoder for the emotions
        enc_layer = nn.TransformerEncoderLayer(d_model = args.feat_dim, nhead = args.nhead_enc, dim_feedforward = 2*args.feat_dim, batch_first = True)
        self.encoder= nn.TransformerEncoder(encoder_layer = enc_layer, num_layers = args.nlayer_enc, norm = layer_norm)

        #last linear layer to go back to vertices coordinates
        self.lin_vertices = nn.Linear(args.feat_dim, args.vertices_dim)



    def forward(self, emotion, vertices, length):
        
        

        #add positional encoding to the emotion embedding
        embedded_emotion = self.pos_enc(embedded_emotion)

        #embed emotion
        embedded_emotion = self.embed_emotion(emotion)

        emotion_features = self.encoder(embedded_emotion)

        
        for i in range(length):
            if i == 0:
                #embed the starting vertices
                emb_vertices = self.embed_vertices(vertices)
                out_vertices = vertices

                #add positional encoding to the vertices embedding
            
            input_vertices = self.pos_enc(emb_vertices)

            #TODO: maybe add a mask for the position in which i find myself in the loop?
            #TODO: implement both tgt_mask and memory_mask

            #decoder
            feature_out = self.decoder(input_vertices, emotion_features, tgt_mask = None, memory_mask = None)
            
            out = self.lin_vertices(feature_out)

            out_vertices = torch.cat((out_vertices, out), 1)

            emb_vertices = torch.cat((emb_vertices, feature_out), 1)

            #TODO: maybe it is wise to do the loss computation inside, to reduce the length of out_vertices?

        return out_vertices



    def predict(self, emotion, vertices, length):
        pass