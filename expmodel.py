import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@torch.no_grad()
class BiasedMask(nn.Module):
    def __init__(self, n_head, max_len: int = 500):
        super().__init__()
        self.n_head = n_head
        self.max_len = max_len
        

    def forward(self, x):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   
            else:                                                 
                closest_power_of_2 = 2**math.floor(math.log2(n)) 
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
        slopes = torch.Tensor(get_slopes(self.n_head))
        bias = torch.arange(start=0, end=self.max_len, step=1).unsqueeze(1).view(-1)
        a = - torch.flip(bias,dims=[0])
        alibi = torch.zeros(self.max_len, self.max_len)
        for i in range(self.max_len):
            alibi[i, :i+1] = a[-(i+1):]
        alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0).expand(self.n_head, -1, -1)
        alibi = alibi.view(self.n_head, self.max_len, self.max_len)
        alibi = alibi.repeat(x.shape[0],1,1)
        mask = (torch.triu(torch.ones(self.max_len, self.max_len)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0) + alibi
        return mask



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)
    

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=10, max_len=500):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def init_tgt_mask(t):
    return torch.triu(torch.ones(t, t) * float('-inf'), diagonal = 1)

def init_mem_mask(t,s):
    mask = torch.ones(t,s)
    for i in range(t):
        mask[i, i+1:] *= float('-inf')
    return mask
    

class ExpModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.device = device
        
        #vertices embedding
        self.embed_vertices = nn.Linear(int(args.vertices_dim)*3, int(args.feat_dim))
        #emotion embedding
        self.embed_emotion = nn.Embedding(int(args.emotion_dim), int(args.feat_dim))
        #positional encoding
        self.pos_enc = PositionalEncoding(int(args.feat_dim), float(args.dropout))
        #self.pos_enc = PeriodicPositionalEncoding(int(args.feat_dim), float(args.dropout), period=10, max_len=61)
        #layernorm for the transformer decoder
        layer_norm = nn.LayerNorm(int(args.feat_dim))
        #transformer decoder
        dec_layer = nn.TransformerDecoderLayer(d_model = int(args.feat_dim), nhead = int(args.nhead_dec), dim_feedforward = 2048, batch_first = True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers = int(args.nlayer_dec), norm = layer_norm)
        #define encoder for the emotions
        enc_layer = nn.TransformerEncoderLayer(d_model = int(args.feat_dim), nhead = int(args.nhead_enc), dim_feedforward = 2048, batch_first = True)
        self.encoder= nn.TransformerEncoder(encoder_layer = enc_layer, num_layers = int(args.nlayer_enc), norm = layer_norm)

        #last linear layer to go back to vertices coordinates
        self.lin_vertices = nn.Linear(int(args.feat_dim), int(args.vertices_dim)*3)

        #bias
        self.bias_mask = BiasedMask(n_head = int(args.nhead_dec), max_len = 500)



    def forward(self, emotion, vertices,template):
        #embed emotion
        embedded_emotion = self.embed_emotion(emotion) #(batch, length, emb_size)


        #add positional encoding to the emotion embedding
        embedded_emotion = self.pos_enc(embedded_emotion) #(batch, length, emb_size)

        emotion_features = self.encoder(embedded_emotion)

        in_vert = vertices[:,:-1,:] - template.unsqueeze(1)
     
        input_vertices = self.embed_vertices(in_vert)

        
        input_vertices = self.pos_enc(input_vertices)
        tgt_mask = self.bias_mask(input_vertices)[:, :input_vertices.shape[1], :input_vertices.shape[1]].clone().detach().to(device = self.device)
        mem_mask = init_mem_mask(input_vertices.shape[1], emotion_features.shape[1]).clone().detach().to(device = self.device)
        feature_out = self.decoder(input_vertices, emotion_features, tgt_mask = tgt_mask, memory_mask = mem_mask)
        out = self.lin_vertices(feature_out)

        out = out + template.unsqueeze(1)

        out = torch.cat((vertices[:,0,:].unsqueeze(1),out),1)
        

        return out 



    def predict(self, emotion, vertices, template, frames):

        #embed emotion
        embedded_emotion = self.embed_emotion(emotion) #(batch, length, emb_size)

        #add positional encoding to the emotion embedding
        embedded_emotion = self.pos_enc(embedded_emotion) #(batch, length, emb_size)

        emotion_features = self.encoder(embedded_emotion)
        for i in range(frames):
            if i == 0:

                emb_vertices = self.embed_vertices(vertices.unsqueeze(1)-template.unsqueeze(1)) #(batch, 1, emb_size)
                input_vertices = self.pos_enc(emb_vertices)
            else:
                input_vertices = self.pos_enc(emb_vertices)

            tgt_mask = self.bias_mask(input_vertices)[:, :input_vertices.shape[1], :input_vertices.shape[1]].clone().detach().to(device = self.device)
            mem_mask = init_mem_mask(input_vertices.shape[1], emotion_features.shape[1]).clone().detach().to(device = self.device)
            
            #out features
            feature_out = self.decoder(input_vertices, emotion_features, tgt_mask = tgt_mask, memory_mask = mem_mask)
            #vertices in vertices dimensions
            out_vertices = self.lin_vertices(feature_out)
            #take last vertices and embed them to feature dimensions
            new_vertices = self.embed_vertices(out_vertices[:,-1,:]).unsqueeze(1)
            #concat embeddings with last embeddings
            emb_vertices = torch.cat((emb_vertices,new_vertices),1)
        
        out_vertices = out_vertices + template.unsqueeze(1)
        out_vertices = torch.cat((vertices.unsqueeze(1),out_vertices),1)

        return out_vertices 
    
