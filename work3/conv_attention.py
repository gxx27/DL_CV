import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Patch(nn.Module):
    def __init__(self, input_size, patch_size=16, emb_dim=768):
        super(Patch, self).__init__()
        
        self.input_size = input_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.patch_shape = (self.input_size-self.patch_size) / self.patch_size + 1
        
    def forward(self, x):
        x = self.conv(x) # batch*3*32*32 -> batch*768*2*2
        x = self.flatten(x) # batch*768*2*2 -> batch*768*4
        x = x.transpose(-1, -2) # batch*768*4 -> batch*4*768
        return x

class ConvAttention(nn.Module):
    def __init__(self, num_head, input_dim, dropout_ratio):
        super(ConvAttention, self).__init__()
        
        self.num_head = num_head # 12
        self.input_dim = input_dim # 768
        self.dropout_ratio = dropout_ratio
        
        self.attention_head = int(input_dim / self.num_head) # 64
        self.all_head = self.num_head * self.attention_head # 768 

        self.drop1 = nn.Dropout(self.dropout_ratio)
        self.drop2 = nn.Dropout(self.dropout_ratio)

        self.w_q = nn.Linear(self.input_dim, self.all_head, bias=False)
        self.w_k = nn.Linear(self.input_dim, self.all_head, bias=False)
        self.w_v = nn.Linear(self.input_dim, self.all_head, bias=False)
        self.output = nn.Linear(self.input_dim, self.input_dim)
        
    def transfer_dim(self, x):
        new_shape = x.size()[:-1] + (self.num_head, self.attention_head) # batch*5*12*64
        x = x.view(*new_shape) # batch*5*768 -> batch*5*12*64
        
        return x.permute(0, 2, 1, 3) # batch*5*12*64 -> batch*12*5*64
    
    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        new_q = self.transfer_dim(q)
        new_k = self.transfer_dim(k)
        new_v = self.transfer_dim(v)
        
        att_scores = torch.matmul(new_q, new_k.transpose(-1,-2)) # batch*12*5*64 * batch*12*64*5
        att_scores /= math.sqrt(self.attention_head) # batch*12*5*5
        
        att_probs = F.softmax(att_scores, dim=-1)
        att_probs = self.drop1(att_probs)
        
        result = torch.matmul(att_probs, new_v) # batch*12*5*64
        result = result.permute(0, 2, 1, 3).contiguous() # batch*12*5*64 -> batch*5*12*64
        
        new_shape = result.size()[:-2] + (self.all_head,) # must include ',' to become a tensor
        result = result.view(*new_shape)
        
        out = self.output(result)
        return self.drop2(out)

class Mlp(nn.Module):
    def __init__(self, dropout_ratio, input_dim, hidden_dim):
        super(Mlp, self).__init__()
        
        self.dropout_ratio = dropout_ratio
        self.drop = nn.Dropout(self.dropout_ratio)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(self.dropout_ratio)
        )
    
    def forward(self, x):
        result = self.mlp(x)
        return self.drop(result)

class Transformer_encoder(nn.Module):
    def __init__(self, dropout_ratio, input_dim, num_head=12, num_layers=12):
        super(Transformer_encoder, self).__init__()
        
        self.dropout_ratio = dropout_ratio
        self.input_dim = input_dim
        self.num_head = num_head
        self.num_layers = num_layers
        
        self.layer_norm1 = nn.LayerNorm(self.input_dim)
        self.attention = ConvAttention(num_head=self.num_head, input_dim=self.input_dim, dropout_ratio=self.dropout_ratio)
        self.dropout1 = nn.Dropout(self.dropout_ratio)
        
        self.layer_norm2 = nn.LayerNorm(self.input_dim)
        self.mlp_block = Mlp(dropout_ratio=self.dropout_ratio, input_dim=self.input_dim, hidden_dim=4*self.input_dim)
        self.dropout2 = nn.Dropout(self.dropout_ratio)

    def forward(self, x):
        for _ in range(self.num_layers):
            norm1 = self.layer_norm1(x) # batch*5*768 -> batch*5*768
            att = self.attention(norm1) # batch*5*768 -> batch*5*768
            drop1 = self.dropout1(att) # batch*5*768 -> batch*5*768
            result1 = x + drop1 # batch*5*768 -> batch*5*768
            
            norm2 = self.layer_norm2(result1) # batch*5*768 -> batch*5*768
            mlp = self.mlp_block(norm2) # batch*5*768 -> batch*5*768
            drop2 = self.dropout2(mlp) # batch*5*768 -> batch*5*768
            result2 = result1 + drop2 # batch*5*768 -> batch*5*768
            x = result2
        return x

class ViT(nn.Module):
    def __init__(self, input_size, batch_size, dropout_ratio, num_class, patch_size=16, emb_dim=768, num_head=12, num_layers=12):
        """
        intput_size: img input size
        batch_size : training batch_size
        dropout_ratio: dropout percent
        num_classes: classfication classes num
        patch_size: patch size
        emb_dim: Patch embedding dim
        num_head: num of multihead attention
        num_layers: num of Transformer Encoder
        """
        super(ViT, self).__init__()
        
        self.input_size = input_size
        self.batch_size = batch_size
        self.dropout_ratio = dropout_ratio
        self.num_class = num_class
        
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.patch_emb = Patch(input_size=self.input_size, patch_size=self.patch_size, emb_dim=self.emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_dim))
        
        img_size = self.patch_emb.patch_shape
        emb_size = int(img_size**2+1)
        self.pos_emb = nn.Parameter(torch.randn(1, emb_size, self.emb_dim))
        self.pos_drop = nn.Dropout(p=self.dropout_ratio)

        self.num_head = num_head
        self.num_layers = num_layers
        self.trans_emb = Transformer_encoder(dropout_ratio=self.dropout_ratio, input_dim=self.emb_dim, num_head=self.num_head, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(self.emb_dim)

        self.mlp_head = nn.Linear(self.emb_dim, self.num_class)
        
    def forward(self, x): # x:batch*3*32*32
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        pos_emb = self.pos_emb.expand(batch_size, -1, -1)

        concat = torch.cat((cls_token, self.patch_emb(x)), dim=1) # batch*3*32*32 -> batch*4*768 -> batch*5*768
        pos = pos_emb + concat # batch*5*768 -> batch*5*768
        drop = self.pos_drop(pos) # batch*5*768 -> batch*5*768
        trans = self.trans_emb(drop) # batch*5*768 -> batch*5*768
        norm = self.norm(trans) # batch*5*768 -> batch*5*768
        cls = norm[:,0] # batch*5*768 -> batch*768
        classification = self.mlp_head(cls) # batch*768 -> batch*10
        
        return classification