## Mask Multihead attention
# Architecture link: https://www.researchgate.net/figure/GPT-2-architecture-Heilbron-et-al-2019_fig1_358654229
import torch 
import torch.nn as nn
import torch.nn.functional as F
from gpt2_config import *

# Multihead attention class
class MultiheadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, context_length:int, dropout:float):
        super().__init__()
        assert d_model%num_heads == 0, "d_model should be divisable by nums_heads"# As we're gonna split embeddings across the heads
        # Let's say, 512/8 = 64 & hence 64 Embeddings/head will get alloted
        self.d_model= d_model
        self.num_heads= num_heads
        self.context_length= context_length
        self.attn_dropout= dropout
        self.head_dim= self.d_model// self.num_heads

        ## attention weight matrixes
        self.qkv =nn.linear(d_model, d_model*3, bias=False)
        self.proj = nn.Linear(d_model, d_model) # to cmprise all infos
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape

        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, 3*d_model)
        qkv=self.qkv(x)
        # to make this available for multiple heads
        # batch_size, seq_len, 3*d_model) -> (batch_size, seq_len, 3, num_heads, head_dims)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        # (batch_size, seq_len, 3, num_heads, head_dims) -> (3, batch_size, num_heads, seq_len, head_dims)
        qkv= qkv.permute(2, 0, 3, 1, 4)

        # 3*(batch_size, num_heads, seq_len, head_dims)
        query, key, value = qkv

        # (batch_size, num_heads, seq_len, head_dims) -> (batch_size, seq_len, num_heads, head_dims) -> (batch_size, seq_len, d_model)
        attn_x = F.scaled_dot_product_attention(
            query= query, 
            key=key,
            value= value,
            attn_mask=None,
            dropout_p = self.attn_dropout,
            is_casual=True
        )

        # concatination 
        concat_attn_x = attn_x.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)
        attn_proj_x = self.proj(concat_attn_x)
        return attn_proj_x


## FeedForward
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff: int, dropout:float):
        super().__init__()
        self.d_model= d_model
        self.d_ff= d_ff
        self.dropout= dropout
        self.fc1= nn.Linear(self.d_model, self.d_ff) # DownProjection
        self.fc2= nn.Linear(self.d_ff, self.d_model) # UpProjection
        self.ff_dropout= dropout
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor):
        # 512 -> 2048 -> 512
        uproj_x= self.fc1(x)
        uproj_x= self.gelu(uproj_x)
        ff_out= self.fc2(uproj_x)
        # ff_out= self.ff_dropout(down_projx)
        return ff_out

# Layer Norm Class
class LayerNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5):
        super().__init__()
        self.eps = eps
        self.d_model= d_model
        self.alpha= nn.parameter(torch.ones(self.d_model))
        self.beta= nn.parameter(torch.ones(self.d_model))
    
    def forward(self, x:torch.Tensor):
        # x: (batch_size, seq_len, d_model)
        # (x- mean)/ std_dev -> alpha * norm_x + beta
        mean= x.mean(-1, keepdim=True)
        var= x.var(dim=-1, unbiased=False, keepdim=True)
        x_normalize = (x- mean)/torch.sqrt(var + self.eps)
        learned_x_normalized = self.alpha*x_normalize + self.beta
        return learned_x_normalized


# Embdeeings Class
class IPEmbeddings(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, dropout:float, context_length:int, device=torch.device):
        super().__init__()
        self.device= device
        self.vocab_size = vocab_size
        self.d_model= d_model # Embedding Dimensions
        self.dropout= dropout
        self.token_embedding= nn.Embedding(self.vocab_size, self.d_model) # For how many words you need TokenEmbeddings
        self.context_length= context_length
        self.positional_embedding= nn.Embedding(self.context_length, self.d_model) # self.context_length coz, we're applying positional encoding on sentence 
        self.dropout_embedding= nn.Dropout(self.dropout)

        # importing from config file
        # self.device= DEVICE
        # self.vocab_size = VOCAB_SIZE
        # self.d_model= D_MODEL # Embedding Dimensions
        # self.dropout= DROPOUT
        # self.token_embedding= nn.Embedding(self.vocab_size, self.d_model) # For how many words you need TokenEmbeddings
        # self.context_length= CONTEXT_LENGTH
        # self.positional_embedding= nn.Embedding(self.context_length, self.d_model) # self.context_length coz, we're applying positional encoding on sentence 
        # self.dropout_embedding= nn.Dropout(self.dropout)

    def forward(self, x:torch.Tensor): # input IDs: x
        # torch.LongTensor([[1, 2, 4, 5, 6,7,8,9,22,24], 
        #           [4, 3, 2, 9,5,7,8,9,22,28]]).shape
        # (2, 10) --> Batch_size, seq_length
        batch_size, seq_length= x.shape

        x_token_embedding= self.token_embedding(x) # size --> seq_length, d_model --> batch_size, seq_length, d_model
        positions= torch.arange(seq_length, device=self.device) # --> [0,1,2,3,4,......, seq_length]
        x_position_embedding= self.positional_embedding(positions) # --> seq_length, d_model

        # Add token Embedding and Postional Embedding
        x_input_embedding= x_token_embedding + x_position_embedding  #--> seq_length, d_model--> batch_size, seq_length, d_model
        return self.dropout(x_input_embedding) #--> batch_size, seq_length, d_model


## GPT BLOCK

class GPTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size= VOCAB_SIZE
        self.d_model= D_MODEL
        self.eps= EPSILON
        self.context_length= CONTEXT_LENGTH
        self.num_heads= N_HEADS
        self.n_layer =N_LAYERS
        self.dropout= DROPOUT
        self.d_ff= D_FF
        self.device=DEVICE

            
        self.layer_norm1= LayerNorm(d_model=self.d_model, eps=self.eps)

        self.multi_head_attention= MultiheadAttention(d_model=self.d_model, num_heads=self.num_heads,
                                                      context_length=self.context_length, dropout= self.dropout)
        
        self.layer_norm2= LayerNorm(d_model=self.d_model, eps=self.eps)
        
        self.feed_forward= FeedForward(d_model= self.d_model, d_ff=self.d_ff, dropout=self.dropout)
        self.add_ln_drop= nn.Dropout(self.dropout)

    def forward(self, x:torch.Tensor):
        before_norm_bef_attn_x= x
        x =self.layer_norm1(x)
        x=self.multi_head_attention(x)
        x= self.add_ln_drop(x)
        x= x + before_norm_bef_attn_x

        before_norm_bef_ffn_x= x
        x =self.layer_norm2(x)
        x=self.feed_forward(x)
        x= self.add_ln_drop(x)
        x= x + before_norm_bef_attn_x
        x= x + before_norm_bef_ffn_x
        return x 

class GPT2WebModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size= VOCAB_SIZE
        self.d_model= D_MODEL
        self.eps= EPSILON
        self.context_length= CONTEXT_LENGTH
        self.num_heads= N_HEADS
        self.n_layer =N_LAYERS
        self.dropout= DROPOUT
        self.d_ff= D_FF
        self.device=DEVICE

        self.input_embedding = IPEmbeddings(self.vocab_size, self.d_model, self.dropout, self.context_length, self.device)

        # self.gpt_block= GPTBlock()
        self.gpt_blocks= nn.Sequential(*[GPTBlock() for _ in range(self.n_layer)])
        self.final_layer_norm= LayerNorm(self.d_model)

        # embedding to Tokens
        self.final_projection=nn.Linear(self.d_model, self.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        input_embeds= self.input_embedding(input_ids)
        gpt_out= self.gpt_blocks(input_embeds)
        gpt_out= self.final_layer_norm(gpt_out)
        logits= self.final_projection(gpt_out)
        return logits

