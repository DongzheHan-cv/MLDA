import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
 
class TransformerModule(nn.Module):
    def __init__(self,in_channels=32,token_len=4, with_pos='learned', with_decoder_pos=None,
                 enc_depth=4, dec_depth=4, dim_head=64, decoder_dim_head=64,decoder_softmax=True):
        super(TransformerModule,self).__init__()
        
        #divide token
        self.token_len = token_len
        self.conv_a = nn.Conv2d(in_channels, self.token_len, kernel_size=1, padding=0, bias=False)
        
        #pos_embedding
        #两种方式：learned，fixed；learned会创建一个可学习的位置嵌入；fixed中位置嵌入是固定不变的，通常以某种规则生成并在训练过程中不发生改变。
        self.with_pos = with_pos
        if self.with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len, in_channels))
        #decoder部分也可以做位置编码
        decoder_pos_size = 256//4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, in_channels, decoder_pos_size, decoder_pos_size))

        #transfomer encoder and decoder
        dim = in_channels
        mlp_dim = 2*dim
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer_encoder = TransformerEncoder(dim=dim, depth=self.enc_depth, heads=8,
                                    dim_head=self.dim_head,mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,heads=8, 
                                    dim_head=self.decoder_dim_head, mlp_dim=mlp_dim, dropout=0,
                                    softmax=decoder_softmax)  

    #单分支前向传播
    def forward(self, x):
        #由backbone得到的X1,X2分别表示灾前灾后的影像
        #tokenzier
        tokens = self._forward_semantic_tokens(x)
        #transformer encoder
        tokens = self._forward_transformer_encoder(tokens)
        #transformer decoder
        x = self._forward_transformer_decoder(x, tokens)
        return x

    # #双分支前向传播
    # def forward(self, x1, x2):
    #     #由backbone得到的X1,X2分别表示灾前灾后的影像
    #     #tokenzier
    #     token1 = self._forward_semantic_tokens(x1)
    #     token2 = self._forward_semantic_tokens(x2)   
    #     tokens = torch.cat([token1, token2], dim=1)
    #     #transformer encoder
    #     tokens = self._forward_transformer_encoder(tokens)
    #     #再分割成两个张量,分别送入decoder
    #     token1, token2 = tokens.chunk(2, dim=1)
    #     #transformer decoder
    #     x1 = self._forward_transformer_decoder(x1, token1)
    #     x2 = self._forward_transformer_decoder(x2, token2)

    #     # feature differencing,做差？求和？
    #     #x = torch.abs(x1 - x2)
    #     x = torch.cat([x1,x2],dim=1)
    #     return x 
    
    #划分tokens
    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        ##将某个维度的大小设为-1时，它表示该维度的大小将根据其他维度的大小和总元素数自动计算。contiguous()是一个用于确保张量存储连续性的函数
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        #bln
        x = x.view([b, c, -1]).contiguous()
        #bcn
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        #blc
        return tokens
    
    #transfomer encoder
    def _forward_transformer_encoder(self, x):
        x += self.pos_embedding
        x = self.transformer_encoder(x)
        return x
    
    #transfomer decoder
    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        #对decoder进行位置编码
        #x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x
  
#==============================“Transfomer Encoder”=============================
class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        #dim：输入数据的特征维度
        self.layers = nn.ModuleList([])
        #depth：编码器中包含的Transformer层数
        for _ in range(depth):
            #heads：多头自注意力机制（Attention）中注意力头的数量。dim_head：每个注意力头的维度。mlp_dim：前馈神经网络（FeedForward）中间层的维度。
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        #mask：输入数据的掩码tensor，用于在自注意力机制中屏蔽一些不需要的位置，以避免信息泄漏。
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

#==============================“Transfomer Decoder”=============================    
class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x

#==============================“Transfomer Help Functions”===========================       
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)

#前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

#交叉注意力机制，用在decoder端
class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # vis_tmp2(out)

        return out

#注意力机制，用在encoder端
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
