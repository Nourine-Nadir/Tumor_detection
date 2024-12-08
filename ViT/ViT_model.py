from einops.layers.torch import Rearrange
from einops import repeat
from torch import nn
import torch as T
import os


def weights_init_(m):
    if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform_(m.weight, gain=1)
        T.nn.init.constant_(m.bias, 0)


class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 8,
                 emb_dim: int = 128):
        self.patch_size = patch_size
        super().__init__()
        self.patch_dim = patch_size * patch_size * in_channels
        self.projections = nn.Sequential(

            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(self.patch_size * self.patch_size),  # Very important improvement
            nn.Linear(self.patch_dim, emb_dim),
            nn.LayerNorm(emb_dim),  # Very important improvement
            # (h * w) / (p1*p2), emb_dim
        )

    def forward(self, x):
        x = self.projections(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 n_heads,
                 dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = nn.MultiheadAttention(embed_dim=dim,
                                         num_heads=n_heads,
                                         dropout=dropout)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        att_output, att_output_weights = self.att(q, k, v)

        return att_output


class PreNorm(nn.Module):
    def __init__(self,
                 dim,
                 fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Sequential):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 dropout: float = 0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )


class ResidualAdd(nn.Module):
    def __init__(self,
                 fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        res = x
        x = self.fn(x)
        x += res
        return x


class Transformer(nn.Module):
    def __init__(self,
                 emb_dim,
                 depth,
                 heads,
                 mlp_dim,
                 dropout):
        super().__init__()
        self.preNorm = PreNorm
        self.residual = ResidualAdd
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                self.residual(self.preNorm(emb_dim, Attention(dim=emb_dim, n_heads=heads, dropout=dropout))),
                self.residual(self.preNorm(emb_dim, FeedForward(dim=emb_dim, hidden_dim=mlp_dim, dropout=dropout))),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(self,
                 lr: float = 1e-3,
                 in_channels: int = 3,
                 img_size: int = 128,
                 patch_size: int = 8,
                 emb_dim: int = 128,
                 n_heads: int = 1,
                 n_layers: int = 1,
                 mlp_dim: int = 256,
                 dropout: float = 0.,
                 out_dim: int = 2, ):
        super().__init__()
        # Attr
        self.channels = in_channels
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim
        self.n_layers = n_layers

        # DEVICE
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        # Xavier glorot weight init
        self.apply(weights_init_)

        # Patching
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=self.patch_size,
                                              emb_dim=self.emb_dim).to(self.device)

        # Learnable params
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            T.randn(1, num_patches + 1, emb_dim)  # + 1 for the cls_token
        )
        # Classification token
        self.cls_token = nn.Parameter(T.rand(1, 1, emb_dim)).to(self.device)

        self.transformer = Transformer(emb_dim=self.emb_dim,
                                       depth=self.n_layers,
                                       heads=self.n_heads,
                                       mlp_dim=self.mlp_dim,
                                       dropout=dropout).to(self.device)
        self.to_latent = nn.Identity().to(self.device)

        self.head = nn.Linear(emb_dim, out_dim).to(self.device)

        self.optimizer = T.optim.AdamW(self.parameters(), lr=lr)
        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(0.995 ** epoch, 1e-2)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, img):
        if img.device != self.device:
            img = img.to(self.device)
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = T.cat([cls_token, x], dim=1)
        x += self.pos_embedding.to(self.device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)

        x = self.head(x)
        return x

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        T.save(self.state_dict(), path)

    def load_model(self, path, map_location=None):
        if map_location is None:
            map_location = self.device  # Use the model's current device
        self.load_state_dict(T.load(path, map_location=map_location, weights_only=True))

    def lr_decay(self):
        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]
