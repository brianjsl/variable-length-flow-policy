from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb, FourierEmbedding
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    """Multilayer perceptron with two hidden layers."""

    def __init__(self, in_dim, hidden_dim, out_dim, act=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g

class CondEncoderBlock(nn.Module):
    def __init__(self,
                cond_dim,
                noise_level_embed_dim,
                num_heads,
                mlp_ratio,
                activation,
                use_rms_norm,
                qkv_bias,
                dropout,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_level_embed_dim = noise_level_embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.activation = activation
        self.use_rms_norm = use_rms_norm
        self.qkv_bias = qkv_bias

        if use_rms_norm:
            self.pre_norm = RMSNorm(cond_dim, eps=1e-6)
            self.post_norm = RMSNorm(cond_dim, eps=1e-6)
        else:
            self.pre_norm = nn.LayerNorm(
                cond_dim, elementwise_affine=False, eps=1e-6
            )
            self.post_norm = nn.LayerNorm(
                cond_dim, elementwise_affine=False, eps=1e-6
            )

        self.attn = Attention(
            dim=cond_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.mlp = MLP(
            in_dim=cond_dim,
            hidden_dim=int(cond_dim * mlp_ratio),
            out_dim=cond_dim,
            act=activation,
            drop=dropout,
        )
        self.ada_ln_chunks = 6
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(noise_level_embed_dim, cond_dim*self.ada_ln_chunks, bias = True)
        )

    def forward(self, x, noise_level_emb):
        x = self.pre_norm(x)
        if noise_level_emb is not None:
            adaln_scale, adaln_shift, adaln_gate, mlp_scale, mlp_shift, mlp_gate = self.ada_ln_modulation(noise_level_emb).chunk(6, dim = 2) # (B, T, D)
            modulated_pre_attn = self.scale_and_shift(x, adaln_shift, adaln_scale)
        else:
            modulated_pre_attn = x
        attn_out = self.attn(modulated_pre_attn)
        if noise_level_emb is not None:
            x = x + attn_out * adaln_gate
            mlp_output = self.mlp(self.scale_and_shift(self.post_norm(x), mlp_shift, mlp_scale))
            x = x + mlp_output * mlp_gate
        else:
            mlp_output = self.mlp(self.post_norm(x))
        return x
    
    def scale_and_shift(self, x, shift, scale):
        result = x * (1 + scale)
        result += shift
        return result

class CondEncoder(nn.Module):
    def __init__(self,
                num_layers,
                cond_dim,
                noise_level_embed_dim,
                num_heads = 8,
                mlp_ratio = 4.0,
                activation = lambda: nn.GELU(approximate="tanh"),
                use_rms_norm = True,
                qkv_bias = True,
                dropout = 0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CondEncoderBlock(
                cond_dim,
                noise_level_embed_dim,
                num_heads,
                mlp_ratio,
                activation,
                use_rms_norm,
                qkv_bias,
                dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, noise_level_emb):
        for layer in self.layers:
            x = layer(x, noise_level_emb)
        return x

class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dim: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=False,
            n_cond_layers: int = 0,
            use_fourier_emb: bool = False
        ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon
        T_cond = 1

        T_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.use_fourier_emb = use_fourier_emb

        # cond encoder
        self.time_emb = FourierEmbedding(n_emb) if use_fourier_emb else SinusoidalPosEmb(n_emb)
        
        self.cond_obs_emb = None
        
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        self.n_cond_layers = n_cond_layers
        encoder_only = False

        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                self.encoder = CondEncoder(
                    num_layers=n_cond_layers,
                    cond_dim=n_emb,
                    noise_level_embed_dim=n_emb,
                    num_heads = n_head,
                    mlp_ratio = 4.0,
                    activation = lambda: nn.GELU(approximate="tanh"),
                    use_rms_norm=False,
                    qkv_bias = True,
                    dropout = p_drop_attn,
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
                self.noise_level_film = nn.Sequential(
                nn.SiLU(), nn.Linear(n_emb, 2 * n_emb, bias = True)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # encoder only BERT
            encoder_only = True

            self.encoder = CondEncoder(
                    num_layers=n_cond_layers,
                    cond_dim=n_emb,
                    noise_level_embed_dim=n_emb,
                    num_heads = n_head,
                    mlp_ratio = 4.0,
                    activation = lambda: nn.GELU(approximate="tanh"),
                    use_rms_norm=False,
                    qkv_bias = True,
                    dropout = p_drop_attn,
            )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            
            S = T_cond
            t, s = torch.meshgrid(
                torch.arange(T),
                torch.arange(S),
                indexing='ij'
            )
            mask = t >= (s-1) # add one dimension since time is the first token in cond
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer('memory_mask', mask)
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            FourierEmbedding if self.use_fourier_emb else SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            nn.SiLU,
            )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, CondEncoder):
            for b in module.layers:
                for m in b.ada_ln_modulation:
                    torch.nn.init.normal_(m, mean=0.0, std=0.02) 
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
        
        if self.n_cond_layers == 0:
            torch.nn.init.normal_(self.noise_level_film[1].weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(self.noise_level_film[1].bias)
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None, 
        noise_levels: Optional[torch.Tensor]=None,
        **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        noise_levels: (B,T)
        output: (B,T,input_dim)
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        if noise_levels is not None:
            noise_levels = self.time_emb(noise_levels)

        # process input
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # BERT
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T+1,n_emb)
            if self.n_cond_layers > 0:
                x = self.encoder(src=x, mask=self.mask, noise_levels=noise_levels)
            else:
                x = self.encoder(x)
                if noise_levels is not None:
                    noise_level_shift, noise_level_scale = self.noise_level_film(noise_levels).chunk(2, dim=2)
                    x = x * (1 + noise_level_scale) + noise_level_shift

            # (B,T+1,n_emb)
            x = x[:,1:,:]
            # (B,T,n_emb)
        else:
            # encoder
            cond_embeddings = time_emb
            cond_obs_emb = self.cond_obs_emb(cond)
            # (B,To,n_emb)
            cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[
                :, :tc, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)

            if self.n_cond_layers > 0:
                x = self.encoder(src=x, mask=self.mask, noise_levels=noise_levels)
            else:
                if noise_levels is not None:
                    noise_level_shift, noise_level_scale = self.noise_level_film(noise_levels).chunk(2, dim=2)
                    x = self.encoder(x) * (1 + noise_level_scale) + noise_level_shift
                else:
                    x = self.encoder(x)

            memory = x
            # (B,T_cond,n_emb)
            
            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask
            )
            # (B,T,n_emb)
        
        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x

def test():
    # GPT with time embedding
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        causal_attn=True,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)
    

    # GPT with time embedding and obs cond
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # GPT with time embedding and obs cond and encoder
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # BERT with time embedding token
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        # causal_attn=True,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)

