import torch
import torch.nn as nn

class PatchEmbedder(nn.Module):
    def __init__(self,
                 input_channel,
                patch_size,
                 hidden_dim
                ):
        super().__init__()
        self.patch_layer = nn.Conv2d(input_channel, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self,x):
        x = self.patch_layer(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        return x

class TimeStepEmbedder(nn.Module):
    def __init__(self,
                 freq_dim,
                 hidden_dim
                ):
        super().__init__()
        #freq
        half_dim = freq_dim//2
        self.freq = 10000**(-2 * torch.arange(0, half_dim)/ half_dim)
        #mlp
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, t):
        t = t.unsqueeze(1)
        freq = self.freq.unsqueeze(0).to(t.device) #(d,) -> (1,d)
        sin_values = torch.sin(t * freq)
        cos_values = torch.cos(t * freq)
        time_embedd = torch.concat([sin_values,cos_values],dim=-1)
        time_embedd = self.mlp(time_embedd)
        return time_embedd

class SelfAttention(nn.Module):
    def __init__(self,
                 heads,
                 head_dim,
                hidden_dim,
                ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.wqkv = nn.Linear(hidden_dim, 3 * heads * head_dim)
        self.wo = nn.Linear(heads * head_dim, hidden_dim)

        self.scale = head_dim ** -0.5

    def forward(self,x):
        
        b, s, d = x.shape
        q, k, v = torch.chunk(self.wqkv(x),3,dim=-1)
        q = q.view(b, s, self.heads, self.head_dim).transpose(1,2)
        k = k.view(b, s, self.heads, self.head_dim).transpose(1,2)
        v = v.view(b, s, self.heads, self.head_dim).transpose(1,2)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1,2).contiguous()
        out = out.view(b,s,self.heads*self.head_dim)
        return self.wo(out)    

class MLP(nn.Module):
    def __init__(self,
                mlp_multiplier,
                hidden_dim
                ):
        super().__init__()

        self.l1 = nn.Linear(hidden_dim, mlp_multiplier * hidden_dim)
        self.l2 = nn.Linear(mlp_multiplier * hidden_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self,x):
        x = self.act(self.l1(x))
        x = self.l2(x)
        return x
        
class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(config['hidden_dim'], elementwise_affine=False)
        self.attn = SelfAttention(config['heads'], config['head_dim'], config['hidden_dim'])
        
        self.norm2 = nn.LayerNorm(config['hidden_dim'], elementwise_affine=False)
        self.mlp = MLP(config['mlp_multiplier'], config['hidden_dim'])
        
        # gamma1, beta1, alpha1 (for attention)
        # gamma2, beta2, alpha2 (for MLP)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config['hidden_dim'], 6 * config['hidden_dim'])
        )

    def forward(self, x, c):
        # x: Image tokens [Batch, Seq_Len, Hidden]
        # c: Conditioning tokens (Time + Label) [Batch, Hidden]
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Apply adaLN: x = x * (1 + scale) + shift
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        # Apply gate: x = x + gate * attention(x)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm)
        
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x

class PositionalEmbedding(nn.Module):
    freq: torch.Tensor
    token_pos: torch.Tensor
    
    def __init__(self,
                 seq_length,
                 hidden_dim
                ):
        super().__init__()
        #freq
        half_dim = hidden_dim//2
        freq = 10000**(-2 * torch.arange(0, half_dim)/ half_dim)
        token_pos = torch.arange(seq_length)

        self.register_buffer('freq', freq)
        self.register_buffer('token_pos', token_pos)
    
    def forward(self):
        freq = self.freq.unsqueeze(0) #(d//2,) -> (1,d//2)
        token_pos = self.token_pos.unsqueeze(1)
        sin_values = torch.sin(token_pos * freq)
        cos_values = torch.cos(token_pos * freq)
        pos_embedd = torch.concat([sin_values,cos_values],dim=-1) #shape -> (seq_len, d)
        return pos_embedd.unsqueeze(0)

class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        x = self.linear(x) 
        return x
        
class DiT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.block_list = nn.ModuleList([DiTBlock(config) for _ in range(config['num_blocks'])])

        self.label_embedding = nn.Embedding(config['num_classes'], config['hidden_dim'])
        self.positional_embedder = PositionalEmbedding((config['image_size']//config['patch_size'])**2, config['hidden_dim'])
        self.patcher = PatchEmbedder(config['input_channel'], config['patch_size'], config['hidden_dim'])
        self.time_embedder = TimeStepEmbedder(config['freq_dim'], config['hidden_dim'])

        self.final_layer = FinalLayer(config['hidden_dim'], config['patch_size'], config['out_channel'])
        
    def unpatchify(self, x):
        """
        Turns the sequence of patches back into an image.
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.config['input_channel']
        p = self.config['patch_size']
        h = w = int(x.shape[1] ** 0.5)
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
        
    def forward(self, x, t, label):

        patches = self.patcher(x)
        time_embedd = self.time_embedder(t)
        class_embedd = self.label_embedding(label)
        pos_embedd = self.positional_embedder()

        c = time_embedd + class_embedd
        x = patches + pos_embedd

        for block in self.block_list:
            x = block(x, c)
        out = self.final_layer(x,c) #(b, h//p*w//p, p*p*c)

        out = self.unpatchify(out)
        return out
        