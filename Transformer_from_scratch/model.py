import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

# Input Embeddings
class IPEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model # Embedding dims
        self.vocab_size = vocab_size
        self.TokenEmbedding= nn.Embedding(self.vocab_size, self.d_model) # Token Embeddings

    def forward(self, x:torch.Tensor):
        # In paper given that, Embedding layer is multiplied with sqrt(d_model)
        return self.TokenEmbedding(x) * torch.sqrt(self.d_model)

# Positional Encodings
# to get the positional information of Tokens/word in the sentence

class PostionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_length:int, drop_out:float):
        super().__init__()
        self.d_model= d_model
        self.seq_len= seq_length
        # self.dropout= drop_out
        self.dropout= nn.Dropout(drop_out)

        # positional matrix (seq_len, d_model)
        pe= torch.zeros(self.seq_len, self.d_model)
        # create vector of shape (seq_len, 1)
        positions= torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1) # (seq_le, 1)
        div_term= torch.exp(torch.arange(0, self.d_model, 2).float()* (-math.log(10000.0)/ self.d_model))
        # apply sin to even position
        pe[:, 0::2] = torch.sin(positions*div_term)
        pe[:, 1::3] =torch.cos(positions*div_term)
        # unsqeez tensors : to add batch dims
        # new shape: (seq_len, d_model) -> (batch, seq_len, d_model): (1, seq_len, d_model)
        pe= pe.unsqueeze(0) ## adding dims at 0'th/ starting  (1, seq_len, d_model)
        self.register_buffer('pe', pe)
        # when you've a tensor that you want to keep inside module as a learned parmas, but  you want to save the file 
    
    def forward(Self, x):
        # here: x is a token embedding
        # add token embedding + positional_encoding
        x= x + (self.pe[:, x.shape[1], : ]).requires_grad(False) # not learnable tensor
        return self.dropout(x)


## Layer Normalization
class LayerNormalization(nn.Module):
    """ To increase the training process smooth & to converge Faster, reduces chances of exploding & vanishing Gradients, usually added before the activation functions
    """
    def __init__(self, eps:float=10**-6):
        super().__init__()
        self.eps= eps
        self.alpha= nn.Parameter(torch.ones(1)) # Multiplied
        self.bias= nn.Parameter(torch.ones(0)) # Added
 
    def forward(self, x:torch.Tensor):
        # (batch_size, seq_len, d_model)
        mean= x.mean(dim=-1, keepdim=True)
        std_dev= x.std(dim=-1, keepdim=True)
        return (x- mean)*self.alpha / (std_dev + self.eps) + self.bias
    
# class FeedForward layer
class Feedforwad(nn.Module):
    """A feed-forward layer in a Transformer is used to apply additional non-linearity and transformation to the data at each position in the sequence after the attention mechanism. It consists of two fully connected layers with an activation function (like ReLU) in between. This allows the model to learn more complex patterns and interactions, enhancing its ability to capture intricate relationships in the data.
    In short, the feed-forward layer helps the Transformer model process information more effectively by adding more depth and non-linearity, which aids in improving its expressiveness and performance. """
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.d_model= d_model
        self.d_ff= d_ff
        self.dropout= dropout
        self.fc1= nn.Linear(self.d_model, self.d_ff) # UpProjection
        self.fc2= nn.Linear(self.d_ff, self.d_model) # DownProjection
        self.ff_dropout= dropout
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        # 512 -> 2048 -> 512
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> # (Batch, seq_len, d_model)
        uproj_x= self.fc1(x)
        uproj_x= self.gelu(uproj_x)
        ff_out= self.fc2(uproj_x)
        ff_out= self.ff_dropout(ff_out)
        return ff_out

# MultiHead Attention
class MutliHeadAttenation(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads= num_heads
 
        assert self.d_model%self.num_heads ==0, "d_model should be divisible by num_heads"
        self.d_k= self.d_model//self.num_heads
        self.w_k = nn.Linear(self.d_model, self.d_model) # Wk
        self.w_q = nn.Linear(self.d_model, self.d_model) # Wq Query weights matrix
        self.w_v = nn.Linear(self.d_model, self.d_model) # Wv Value weights matrix
        self.w_o = nn.Linear(self.d_model, self.d_model)
        self.dropout= nn.Dropout(dropout)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask, dropout: nn.Dropout):
        d_k= query.shape[-1]
        # (Batch, h , seq_len, d_k) --> (Batch, h, seq_len, seq_len)
        attention_scores= (query @ key.transpose(-2, -1))/ math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        
        attention_scores= attention_scores.softmax(dim=-1) # batch, h , seq_len, seq_len

        if dropout is not None:
            attention_scores= dropout(attention_scores)

        return (attention_scores @ value), attention_scores

      
    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask):
        # q,k,v : (batch, seq_len, d_model)
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key= self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value= self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # devide the matrixes across num_heads (self.d_k)
        # (batch, seq_len, d_model) --> (batch, seq_len, num_heads, d_k) --> (batch, num_heads ,seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1,2)
        x, self.attention_scores= MutliHeadAttenation.attention(query, key, value, mask, self.dropout)

        # (Batch, h, Seq_len, d_k) -> (Batch,Seq_len, h, d_k) -> (Batch, Seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k )
        # (Batch, Seq_len, d_model) ----> (Batch, Seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout:float ) -> None:
        super().__init__()
        self.dropout= nn.Dropout(dropout)
        self.norm= LayerNormalization()

    def forward(self, x, sublayer):
        return x+ self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MutliHeadAttenation, feedForward: Feedforwad, dropout:float) -> None:
        super().__init__()
        self.self_attentionBlock= self_attention_block
        self.feed_forward_block= feedForward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range(2)])
    
    def forward(self, x , src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attentionBlock(x,x,x,src_mask))
        x= self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList ) -> None:
        super().__init__()
        self.layers= layers
        self.norm = LayerNormalization()

    def forward(self,x , mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)