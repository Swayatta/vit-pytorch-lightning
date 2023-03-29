import torch
import numpy as np
import torch.nn as nn
from einops import rearrange, reduce, repeat

class Patchify():
    def __init__(self, height_of_patch, width_of_patch) -> None:
        self.h = height_of_patch
        self.w = width_of_patch
    def patchify(self, image):
        '''
        Takes an image tensor as input. Breaks the image into patches and returns the flattened
        patches in the form of a sequence.
        '''
        batch, seq_len, height_of_image, width_of_image, channel = image.shape
        assert height_of_image % self.h == 0 and width_of_image % self.w == 0, 'Image dimensions must be divisible by patch dimensions'
        
        num_patches_height = height_of_image // self.h
        num_patches_width  = width_of_image // self.w
        
        return rearrange(image, 'b n (h1 h) (w1 w) c-> b (n h1 w1) h w c', h1= num_patches_height, w1= num_patches_width)
    

class Linear_Projection_and_Add_CLS():
    def __init__(self, batch_size, seq_len, height_of_patch, width_of_patch, channel, dim) -> None:
        

        total_patch_dim = height_of_patch * width_of_patch * channel
        self.linear_mapper = nn.Linear(total_patch_dim, dim)
        self.cls = nn.Parameter(torch.randn(batch_size,1,dim))

    def project(self, patches):
        '''
        Input : patches (Tensor) - The sequence of patches for an image
        Output : linear maps (Tensor) - The projection of the sequence of patches in a vector space
        '''
        patches = torch.flatten(patches, start_dim = 2, end_dim = -1)
        linear_maps = self.linear_mapper(patches)
        return linear_maps
    
    def add_cls(self, linear_maps):
        '''
        Input : linear_maps (Tensor) - The projection of the sequence of patches in a vector space
        Output : linear maps with cls (Tensor) - CLS token added to the linear projection. The token
                    is added to the beginning of the projected sequence
        '''
        # adding CLS token

        linear_maps_with_cls = torch.cat([self.cls, linear_maps], dim = 1) # adding CLS token to the beginning of each linearly projected sample in the batch
        return linear_maps_with_cls
    
class PositionalEncoding():
    def __init__(self, seq_len, dim) -> None:
        self.seq_len = seq_len
        self.d = dim
        self.n = 10000
    def getPositionEncoding(self):
        '''
        Generates positional encoding using sinusoidal waves
        '''
        P = np.zeros((self.seq_len, self.d))
        for k in range(self.seq_len):
            for i in np.arange(int(self.d/2)):
                denominator = np.power(self.n, 2*i/self.d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return torch.tensor(P, dtype = float)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, attention_head_size, num_attention_heads) -> None:
        super().__init__()
        total_head_size = num_attention_heads * attention_head_size
        self.attention_head_size = attention_head_size
        self.num_attention_heads = num_attention_heads
        self.query = nn.Linear(dim , total_head_size )
        self.key = nn.Linear(dim , total_head_size)
        self.value = nn.Linear(dim , total_head_size)

        self.softmax = nn.Softmax(dim = -1)
        self.dense = nn.Linear(total_head_size, dim)

    def project(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_x_shape)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        q,k,v = self.project(q), self.project(k), self.project(v) # Batch, seq_len, num_of_attn_heads, head_dim
        attention_scores = self.softmax(torch.matmul(q, k.transpose(-1,-2)))    # Batch, seq_len, num_of_attn_heads, head_dim
        context_vector = torch.matmul(attention_scores, v)  # Batch, seq_len, num_of_attn_heads, head_dim
        context_vector = rearrange(context_vector, 'b n num_att h -> b n (num_att h)')  # batch, seq_len, total_dim = num_of_attn_heads * head_dim
        output = self.dense(context_vector)     # batch, seq_len, dim
        return output            


class Transfomer(nn.Module):
    def __init__(self, num_blocks, dim, mlp_dim, attention_head_size, num_attention_heads) -> None:
        super().__init__()
        # assert input_dim == attention_head_size
        self.norm1 = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([MultiHeadAttention(dim, attention_head_size, num_attention_heads) for _ in range(num_blocks)])
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim,dim)
            )

    def forward(self,x):
        for layer in self.layers:
            x = x+layer(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, batch_size, height_of_patch, width_of_patch,
                  channel, num_blocks, dim, mlp_dim, attention_head_size, num_attention_heads, num_of_labels, seq_len = 1) -> None:
        super().__init__()
        self.patchify = Patchify(height_of_patch, width_of_patch) ### TBD
        self.linear_and_cls = Linear_Projection_and_Add_CLS(batch_size, seq_len, height_of_patch, width_of_patch, channel, dim)
        self.pos_enc = PositionalEncoding(seq_len, dim)
        self.transformer = Transfomer(num_blocks, dim, mlp_dim, attention_head_size, num_attention_heads)
        self.linear = nn.Linear(dim, num_of_labels)

    
    
    def forward(self, image):
        patches = self.patchify.patchify(image)
        linear_projections = self.linear_and_cls.project(patches)
        linear_projections_with_cls = self.linear_and_cls.add_cls(linear_projections)
        positional_encoding = self.pos_enc.getPositionEncoding()
        input_to_transformer = (linear_projections_with_cls + positional_encoding).float()
        output_from_transformer = self.transformer(input_to_transformer)  #B, seq_len, dim
        output_from_transformer = output_from_transformer[:,0,:]    # cls pooling
        output = self.linear(output_from_transformer)
        return output