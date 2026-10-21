import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, p=64, g=64):
        super(HetConv, self).__init__()
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             groups=g, padding=kernel_size // 2, stride=stride)
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p, stride=stride)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


class SoftmaxTokenPooling(nn.Module):
    def __init__(self, dim, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.queries = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.scale = dim ** -0.5
        
    def forward(self, x):
        B, N, D = x.shape
        queries = self.queries.expand(B, -1, -1)
        attn = torch.bmm(queries, x.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, x)
        return out


class LiDARStar(nn.Module):
    def __init__(self, FM, NCLidarMS):
        super(LiDARStar, self).__init__()
        self.dim = FM * 4
        
        self.lidarConv = nn.Sequential(
            nn.Conv2d(NCLidarMS, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.GELU(),
            nn.Conv2d(FM * 4, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.GELU()
        )
        
        self.token_pool = SoftmaxTokenPooling(dim=self.dim, num_tokens=15)
        
    def forward(self, x2):
        B = x2.shape[0]
        x2 = x2.reshape(B, -1, 11, 11)
        x2 = self.lidarConv(x2)
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        x2 = self.token_pool(x2)
        return x2



class MultiScaleSpectralConv_DINO(nn.Module):
    def __init__(self, in_channels, hidden_dim=32, use_dilation=False):
        super().__init__()
        self.use_dilation = use_dilation
        
        if use_dilation:
            self.conv_local = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1, dilation=1)
            self.conv_medium = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=3, dilation=3)
            self.conv_global = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=6, dilation=6)
        else:
            self.conv_local = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
            self.conv_medium = nn.Conv1d(1, hidden_dim, kernel_size=7, padding=3)
            self.conv_global = nn.Conv1d(1, hidden_dim, kernel_size=11, padding=5)
        
        self.norm_l = nn.LayerNorm(hidden_dim)
        self.norm_m = nn.LayerNorm(hidden_dim)
        self.norm_g = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)
        
        local = self.norm_l(self.conv_local(x_flat).transpose(1, 2))
        medium = self.norm_m(self.conv_medium(x_flat).transpose(1, 2))
        glob = self.norm_g(self.conv_global(x_flat).transpose(1, 2))
        
        return local, medium, glob, (B, C, H, W)



class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class DeepCrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1_hsi = nn.LayerNorm(dim)
        self.norm1_lidar = nn.LayerNorm(dim)
        self.norm2_hsi = nn.LayerNorm(dim)
        self.norm2_lidar = nn.LayerNorm(dim)
        
        self.cross_attn_h2l = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_l2h = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.ffn_hsi = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout)
        )
        self.ffn_lidar = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout)
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.3))  # 残差系数
    
    def forward(self, hsi, lidar):
        hsi_norm = self.norm1_hsi(hsi)
        lidar_norm = self.norm1_lidar(lidar)
        
        hsi_out, _ = self.cross_attn_h2l(hsi_norm, lidar_norm, lidar_norm)
        lidar_out, _ = self.cross_attn_l2h(lidar_norm, hsi_norm, hsi_norm)
        
        hsi = hsi + self.alpha * hsi_out
        lidar = lidar + self.alpha * lidar_out
        
        hsi = hsi + self.ffn_hsi(self.norm2_hsi(hsi))
        lidar = lidar + self.ffn_lidar(self.norm2_lidar(lidar))
        
        return hsi, lidar


class FeedForwardModule(nn.Module):
    def __init__(self, dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, dim, kernel_size=15, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pointwise_expand = nn.Linear(dim, dim * 2)
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.conv_norm = nn.LayerNorm(dim)
        self.activation = nn.SiLU()
        self.pointwise_reduce = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = self.pointwise_expand(x)           # [B, T, 2D]
        x1, x2 = x.chunk(2, dim=-1)            # GLU gating
        x = x1 * torch.sigmoid(x2)             # [B, T, D]
        x = x.transpose(1, 2)                  # [B, D, T]
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)                  # [B, T, D]
        x = self.conv_norm(x)
        x = self.activation(x)
        x = self.pointwise_reduce(x)
        return self.dropout(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, kernel_size=15, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, ffn_dim, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvModule(dim, kernel_size, dropout)
        self.ffn2 = FeedForwardModule(dim, ffn_dim, dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.attn_drop(attn_out)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class ConformerBackbone(nn.Module):
    def __init__(
        self,
        embed_dim=64,
        depth=2,
        num_heads=4,
        ffn_dim=256,
        out_dim=384,
        depthwise_conv_kernel_size=15,
        dropout=0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.blocks = nn.ModuleList([
            ConformerBlock(embed_dim, num_heads, ffn_dim,
                           depthwise_conv_kernel_size, dropout)
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        tokens = rearrange(x, 'b c h w -> b (h w) c')    
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm_out(tokens)
        pooled = tokens.mean(dim=1)                      
        return self.proj(pooled)                      


class Compute_BridgeAttnGatedRegConformer(nn.Module):
    def __init__(
        self,
        Classes,
        embed_dim=64,
        use_aux_loss=True,
        gate_last_bias=-2.0,
        conformer_depth=2,
        conformer_heads=4,
        conformer_ffn_dim=256,
        conformer_conv_kernel=15,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_aux_loss = use_aux_loss

        self.cross_modal = DeepCrossModalAttention(
            dim=embed_dim, num_heads=4, dropout=0.1)
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate_proj[2].bias, gate_last_bias)

        self.conformer_backbone = ConformerBackbone(
            embed_dim=embed_dim,
            depth=conformer_depth,
            num_heads=conformer_heads,
            ffn_dim=conformer_ffn_dim,
            out_dim=384,
            depthwise_conv_kernel_size=conformer_conv_kernel,
            dropout=0.1,
        )
        self.hidden_size = 384

        self.hsi_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.lidar_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))
        nn.init.trunc_normal_(self.hsi_type_embed, std=0.02)
        nn.init.trunc_normal_(self.lidar_type_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(0.15)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.head = MLPHead(self.hidden_size, self.hidden_size, Classes,
                            dropout=0.3)

        if use_aux_loss:
            self.aux_head = nn.Linear(embed_dim, Classes)

    def forward(self, hsi_tokens, lidar_tokens):
        aux_out = None
        if self.use_aux_loss and self.training:
            aux_out = self.aux_head(hsi_tokens.mean(dim=1))

        hsi_tokens = hsi_tokens + self.hsi_type_embed
        tokens = torch.cat([hsi_tokens, lidar_tokens + self.lidar_type_embed], dim=1)
        tokens = tokens + self.pos_embed

        hsi_part = tokens[:, :49, :]
        lidar_part = tokens[:, 49:, :]
        enhanced_hsi, enhanced_lidar = self.cross_modal(hsi_part, lidar_part)

        gate_h = self.gate_proj(torch.cat([hsi_part, enhanced_hsi], dim=-1))
        gate_l = self.gate_proj(torch.cat([lidar_part, enhanced_lidar], dim=-1))
        hsi_out = hsi_part + gate_h * (enhanced_hsi - hsi_part)
        lidar_out = lidar_part + gate_l * (enhanced_lidar - lidar_part)
        tokens = torch.cat([hsi_out, lidar_out], dim=1)

        gate_reg = (gate_h.mean() + gate_l.mean()) * 0.5

        tokens = self.dropout(tokens)
        x = rearrange(tokens, 'b (h w) c -> b c h w', h=8, w=8)
        features = self.conformer_backbone(x)

        main_out = self.head(self.norm(features))
        if self.training and aux_out is not None:
            return main_out, aux_out, gate_reg
        return main_out


class SpectralDenoisingHead(nn.Module):
    def __init__(self, num_bands, rank=20, alpha_init=0.1):
        super().__init__()
        self.num_bands = num_bands
        self.rank = min(rank, num_bands // 2)
        self.down = nn.Linear(num_bands, self.rank, bias=False)
        self.up = nn.Linear(self.rank, num_bands, bias=False)
        self.norm = nn.LayerNorm(num_bands)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        nn.init.orthogonal_(self.down.weight)
        self.up.weight.data = self.down.weight.data.t().clone()

    def forward(self, x):
        x_t = x.transpose(1, 2)        
        x_norm = self.norm(x_t)
        denoised = self.up(self.down(x_norm))
        out = x_t + self.alpha * (denoised - x_norm)
        return out.transpose(1, 2)    


class DINOStyleCrossAttentionSoft(nn.Module):

    def __init__(self, q_dim, kv_dim, num_heads=4, temperature=2.0):
        super().__init__()
        assert q_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.temperature = temperature
        self.scale = (self.head_dim ** -0.5) / temperature

        self.q = nn.Linear(q_dim, q_dim)
        self.k = nn.Linear(kv_dim, q_dim)
        self.v = nn.Linear(kv_dim, q_dim)
        self.proj = nn.Linear(q_dim, q_dim)
        self.norm = nn.LayerNorm(q_dim)

    def forward(self, query, key_value):
        B, N, C = query.shape
        M = key_value.shape[1]

        q = self.q(query).reshape(
            B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(key_value).reshape(
            B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(key_value).reshape(
            B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.norm(query + self.proj(out))


class HSIStar_SDH_SoftAnchor(nn.Module):

    def __init__(self, FM, NC, hidden_dim=32, detach_lidar=True,
                 use_dilation=False, sdh_rank=20, sdh_alpha_init=0.1,
                 anchor_temp=2.0, use_sdh=True):
        super().__init__()
        self.dim = FM * 4
        self.hidden_dim = hidden_dim
        self.detach_lidar = detach_lidar
        self.use_dilation = use_dilation
        self.use_sdh = use_sdh

        if use_sdh:
            self.sdh = SpectralDenoisingHead(
                num_bands=NC, rank=sdh_rank, alpha_init=sdh_alpha_init)
        else:
            self.sdh = None

        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0, 1, 1), stride=1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            HetConv(
                8 * (NC - 8), FM * 4, p=1,
                g=(FM * 4) // 4 if (8 * (NC - 8)) % FM == 0 else (FM * 4) // 8
            ),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU()
        )

        self.mssc = MultiScaleSpectralConv_DINO(
            FM * 4, hidden_dim=hidden_dim, use_dilation=use_dilation)
        self.lidar_proj = nn.Linear(FM * 4, hidden_dim)

        self.ca_local = DINOStyleCrossAttentionSoft(
            hidden_dim, hidden_dim, num_heads=4, temperature=anchor_temp)
        self.ca_medium = DINOStyleCrossAttentionSoft(
            hidden_dim, hidden_dim, num_heads=4, temperature=anchor_temp)
        self.ca_global = DINOStyleCrossAttentionSoft(
            hidden_dim, hidden_dim, num_heads=4, temperature=anchor_temp)

        self.scale_weight = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )

        self.proj_out = nn.Linear(hidden_dim, FM * 4)
        self.token_pool = SoftmaxTokenPooling(dim=self.dim, num_tokens=49)

    def forward(self, x1, lidar_tokens):
        B = x1.shape[0]
        if self.use_sdh:
            x1 = self.sdh(x1)

        x1 = x1.reshape(B, -1, 11, 11)
        x1 = x1.unsqueeze(1)
        x1 = self.conv5(x1)
        x1 = x1.reshape(B, -1, 11, 11)
        hsi_feat = self.conv6(x1)

        local, medium, glob, shape = self.mssc(hsi_feat)
        _, C, H, W = shape

        if self.detach_lidar:
            lidar_for_dino = lidar_tokens.detach()
        else:
            lidar_for_dino = lidar_tokens

        lidar_hidden = self.lidar_proj(lidar_for_dino)
        lidar_expanded = lidar_hidden.unsqueeze(1).expand(-1, H * W, -1, -1)
        lidar_expanded = lidar_expanded.reshape(
            B * H * W, 15, self.hidden_dim)

        local_out = self.ca_local(local, lidar_expanded)
        medium_out = self.ca_medium(medium, lidar_expanded)
        global_out = self.ca_global(glob, lidar_expanded)

        local_pool = local_out.mean(dim=1)
        medium_pool = medium_out.mean(dim=1)
        global_pool = global_out.mean(dim=1)

        weights = self.scale_weight(torch.cat(
            [local_pool, medium_pool, global_pool], dim=-1))
        fused = (weights[:, 0:1] * local_pool +
                 weights[:, 1:2] * medium_pool +
                 weights[:, 2:3] * global_pool)

        fused_out = self.proj_out(fused).reshape(
            B, H, W, self.dim).permute(0, 3, 1, 2)
        hsi_feat = hsi_feat + fused_out * 0.1

        hsi_tokens = rearrange(hsi_feat, 'b c h w -> b (h w) c')
        hsi_tokens = self.token_pool(hsi_tokens)
        return hsi_tokens


class MFT_ConformerSDHSoftAnchor(nn.Module):
    def __init__(
        self,
        FM,
        NC,
        NCLidarMS,
        Classes,
        detach_lidar=True,
        use_dilation=False,
        use_aux_loss=True,
        gate_last_bias=-2.0,
        conformer_depth=2,
        conformer_heads=4,
        conformer_ffn_dim=256,
        conformer_conv_kernel=15,
        sdh_rank=20,
        sdh_alpha_init=0.1,
        anchor_temp=2.0,
    ):
        super().__init__()
        self.lidar_star = LiDARStar(FM, NCLidarMS)
        self.hsi_star = HSIStar_SDH_SoftAnchor(
            FM, NC, hidden_dim=32,
            detach_lidar=detach_lidar,
            use_dilation=use_dilation,
            sdh_rank=sdh_rank,
            sdh_alpha_init=sdh_alpha_init,
            anchor_temp=anchor_temp,
        )
        self.compute_star = Compute_BridgeAttnGatedRegConformer(
            Classes,
            embed_dim=FM * 4,
            use_aux_loss=use_aux_loss,
            gate_last_bias=gate_last_bias,
            conformer_depth=conformer_depth,
            conformer_heads=conformer_heads,
            conformer_ffn_dim=conformer_ffn_dim,
            conformer_conv_kernel=conformer_conv_kernel,
        )

    def forward(self, x1, x2):
        lidar_tokens = self.lidar_star(x2)
        hsi_tokens = self.hsi_star(x1, lidar_tokens)
        return self.compute_star(hsi_tokens, lidar_tokens)
