from typing import Dict, Tuple, Optional
import dill
import math
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import wandb
import numpy as np
import omegaconf

def dct_2d(x):
    """
    2D Discrete Cosine Transform using matrix multiplication.
    """
    N = x.shape[-2]
    M = x.shape[-1]
    
    # Create DCT transform matrices
    u = torch.arange(N).float().view(1, -1).to(x.device)  # Row indices
    v = torch.arange(M).float().view(-1, 1).to(x.device)  # Column indices
    
    dct_matrix_row = torch.cos((2 * u + 1) * v * torch.pi / (2 * N))
    dct_matrix_col = torch.cos((2 * v + 1) * u * torch.pi / (2 * M))
    
    # Apply DCT transformation
    dct_out = torch.matmul(dct_matrix_row, x)
    dct_out = torch.matmul(dct_out, dct_matrix_col)
    
    return dct_out

def compress_image_with_dct(x):
    x = x.cuda()
    # x: (batch_size, 3, 70, 70)
    batch_size = x.shape[0]
    
    # Apply DCT to each channel
    dct_r = dct_2d(x[:, 0, :, :])
    dct_g = dct_2d(x[:, 1, :, :])
    dct_b = dct_2d(x[:, 2, :, :])
    
    # Keep top-left corner coefficients (8x8 from each channel)
    top_r = dct_r[:, :4, :4].flatten(start_dim=1)  # Shape: (batch_size, 64)
    top_g = dct_g[:, :4, :4].flatten(start_dim=1)  
    top_b = dct_b[:, :4, :4].flatten(start_dim=1)  
    
    # Concatenate and truncate to 64 elements
    compressed_vector = torch.cat([top_r, top_g, top_b], dim=1)#[:, :64]
    
    return compressed_vector

class DiffusionTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            use_flow_matching: bool = False,
            input_pertub: float = 0,
            fm_tsampler: str = "uniform", 
            num_inference_steps=None,
            train_diffusion_n_samples: int = 1,
            use_embed_if_present=True,
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            pred_action_steps_only=False,
            past_action_pred=False,
            obs_encoder_dir=None,
            obs_encoder_freeze=False,
            past_steps_reg=-1,
            hist_guidance: Optional[omegaconf.DictConfig] = None,
            use_fourier_emb: bool = False,
            **kwargs):
        super().__init__()

        self.use_flow_matching = use_flow_matching
        self.fm_tsampler = fm_tsampler
        self.input_pertub = input_pertub
        self.hist_guidance = hist_guidance

        if self.fm_tsampler == "beta":
            # Following https://arxiv.org/abs/2410.24164 pi-0 Appendix B
            # to upsample timesteps near 1 (high noise level).
            self.tsampler = torch.distributions.beta.Beta(1.5, 1.0)

        self.past_action_pred = past_action_pred
        self.use_embed_if_present = use_embed_if_present
        self.past_steps_reg = past_steps_reg

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim
        output_dim = input_dim
        cond_dim = obs_feature_dim

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            n_cond_layers=n_cond_layers,
            use_fourier_emb = use_fourier_emb
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.use_fourier_emb = use_fourier_emb # for flow
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if obs_encoder_dir:
            print(f"loading encoder from {obs_encoder_dir}")
            # obs_encoder_path = pathlib.Path(obs_encoder_dir)

            payload = torch.load(open(obs_encoder_dir, "rb"), pickle_module=dill)
            payload['state_dicts']['obs_encoder'] = {}
            for key, value in payload['state_dicts']['model'].items():
                if key.startswith("obs_encoder"):
                    # split keys
                    payload['state_dicts']['obs_encoder'][key[len('obs_encoder.'):]] = value
            self.obs_encoder.load_state_dict(payload['state_dicts']['obs_encoder'], **kwargs)
        if obs_encoder_freeze:
            print("freezing encoder")
            for param in self.obs_encoder.parameters():
                param.requires_grad = False


        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps

        self.num_inference_steps = num_inference_steps
        self.train_diffusion_n_samples = train_diffusion_n_samples

        if self.hist_guidance.training.type == "random_independent":
            print(f"Training with random independent history length")
        elif self.hist_guidance.training.type == "binary_dropout":
            print(f"Training with binary dropout history length")
        elif self.hist_guidance.training.type == "fixed":
            print(f"Training with fixed history length: {self.hist_guidance.training.training_length if self.hist_guidance.training.training_length != 'FULL' else self.n_obs_steps}")
        elif self.hist_guidance.training.type == "full":
            print(f"Training with full history length")
        else:
            raise ValueError(f"Invalid hist_guidance training type: {self.hist_guidance.training.type}")

        if self.hist_guidance.sampling.type == "fixed" or self.hist_guidance.sampling.type == "fixed_stabilization" or self.hist_guidance.sampling.type == "pyramid":
            print(f"Sampling with fixed history length: {self.hist_guidance.sampling.hist_len}")
        elif self.hist_guidance.sampling.type == "average": 
            print(f"Sampling with history lengths averaged between {self.hist_guidance.sampling.hist_len} with weights {self.hist_guidance.sampling.weights}")
            assert self.hist_guidance.sampling.weights is not None
        elif self.hist_guidance.sampling.type == "full":
            print(f"Sampling with full history length")
        elif self.hist_guidance.sampling.type == "argmin":
            print(f"Sampling with full history length and {self.hist_guidance.sampling.n_samples} samples")
            assert self.hist_guidance.sampling.n_samples is not None
        elif self.hist_guidance.sampling.type == "dynamic":
            assert self.hist_guidance.training.type == "random_independent" or self.hist_guidance.training.type == "binary_dropout"
            assert self.hist_guidance.sampling.n_samples is not None
            print(f"Sampling with dynamic history length")
        else:
            raise ValueError(f"Invalid hist_guidance sampling type: {self.hist_guidance.sampling.type}")
    
    # ========= inference  ============
    def conditional_sample(self, 
            shape,
            cond=None, 
            generator=None, 
            obs_mask = None,
            hist_len = None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler
        B, T, D = shape

        if self.hist_guidance.sampling.type == "argmin":
            bsz = B * self.hist_guidance.sampling.n_samples
        elif self.hist_guidance.sampling.type == "dynamic":
            bsz = B * (self.hist_guidance.sampling.n_samples + 1)
        else:
            bsz = B
        
        trajectory = torch.randn(
            size=(bsz, T, D), 
            dtype=cond.dtype,
            device=cond.device,
            generator=generator)
        
        if self.hist_guidance.training.type == "random_independent" or self.hist_guidance.training.type == "binary_dropout":
            if self.hist_guidance.sampling.type == "pyramid":
                noise_levels = self._generate_pyramid_scheduling_matrix(B * len(hist_len), self.hist_guidance.sampling.max_noise_level, self.n_obs_steps)
            else:
                noise_levels = torch.zeros(B * len(hist_len), self.n_obs_steps, device = self.device) 
        else:
            noise_levels = None
        
        flattened_obs_mask = obs_mask.view(-1, *obs_mask.shape[2:])

        if self.hist_guidance.training.type == "random_independent" or self.hist_guidance.training.type == "binary_dropout":
            if hasattr(self.hist_guidance.sampling, "mask_level"): 
                noise_levels[~flattened_obs_mask] = self.hist_guidance.sampling.mask_level # unpredicted tokens are set to "full noise"
            else:
                noise_levels[~flattened_obs_mask] = 1.0 # unpredicted tokens are set to "full noise"
        
        cond = cond.view(-1, *cond.shape[2:])

        # noise as masking
        if self.hist_guidance.training.type == "random_independent" or self.hist_guidance.training.type == "binary_dropout":
            cond = self._apply_noising(cond, noise_levels)

        # for stabilization we just set the noise level to the stabilization level for the predicted tokens (no actual noising)

        if self.hist_guidance.training.type == "random_independent" or self.hist_guidance.training.type == "binary_dropout":
            if self.hist_guidance.sampling.type == "fixed_stabilization":
                noise_levels[flattened_obs_mask] = self.hist_guidance.sampling.stabilization_level
            if not self.use_fourier_emb and noise_levels is not None:
                noise_levels = noise_levels * self.noise_scheduler.config.num_train_timesteps

        noise_levels = torch.cat([torch.zeros((noise_levels.shape[0], 1), device = self.device, dtype = torch.float32), noise_levels], dim = 1) # add a zero noise level for the timestep

        if self.use_flow_matching:
            timesteps = torch.linspace(
                1,
                0,
                self.num_inference_steps + 1,
                device = self.device
            )[:-1]

            for t in timesteps:
                if not self.use_fourier_emb:
                    t = (
                        t * self.noise_scheduler.config.num_train_timesteps
                    ).float()

                actions = trajectory

                if self.hist_guidance.sampling.type not in ["dynamic"]:
                    actions = actions.repeat(len(hist_len), 1, 1)

                model_output = model(
                    trajectory,
                    t,
                    cond,
                    noise_levels
                )

                if self.hist_guidance.sampling.type == "average":
                    weights = torch.tensor(self.hist_guidance.sampling.weights, device=trajectory.device) # (len(hist_len),)
                    weights = weights / weights.sum()
                else:
                    weights = torch.tensor([1.], device=trajectory.device)

                if self.hist_guidance.sampling.type != "dynamic":
                    model_output = model_output.view(B, len(hist_len), *model_output.shape[1:]) # (B, len(hist_len), Ta, D)
                    weights = weights[None, :, None].repeat(1, 1, T) 
                    weights = weights / weights.sum(dim = 1, keepdim = True)
                    dim_diff = model_output.ndim - weights.ndim
                    weights = weights.view(*weights.shape, *([1] * dim_diff))
                    model_output = (model_output * weights).sum(dim=1)

                trajectory = (
                    trajectory - model_output / self.num_inference_steps
                )
        else:
            # set step values
            scheduler.set_timesteps(self.num_inference_steps)

            for t in scheduler.timesteps:
                timesteps = t.to(trajectory.device)

                if self.use_fourier_emb:
                    timesteps = timesteps / self.noise_scheduler.config.num_train_timesteps
                
                if self.hist_guidance.sampling.type not in ["dynamic"]:
                    trajectory= trajectory.repeat(len(hist_len), 1, 1)

                model_output = model(trajectory, timesteps, cond, noise_levels)

                if self.hist_guidance.sampling.type == "average":
                    weights = torch.tensor(self.hist_guidance.sampling.weights, device=trajectory.device) # (len(hist_len),)
                    weights = weights / weights.sum()
                else:
                    weights = torch.tensor([1.], device=trajectory.device)

                if self.hist_guidance.sampling.type != "dynamic":
                    model_output = model_output.view(B, len(hist_len), *model_output.shape[1:]) # (B, len(hist_len), Ta, D)
                    weights = weights[None, :, None].repeat(1, 1, T)
                    weights = weights / weights.sum(dim = 1, keepdim = True)
                    dim_diff = model_output.ndim - weights.ndim
                    weights = weights.view(*weights.shape, *([1] * dim_diff))
                    model_output = (model_output * weights).sum(dim=1)

                trajectory = scheduler.step(
                    model_output, t, trajectory, 
                    generator=generator,
                    **kwargs
                ).prev_sample
        
        if self.hist_guidance.sampling.type == "dynamic":
            trajectory = trajectory.view(len(hist_len), B, *trajectory.shape[1:])
            full_trajectory = trajectory[-1].unsqueeze(0).repeat(len(hist_len) - 1, 1, 1, 1)
            trajectory = trajectory[:-1]
            l2_loss = F.mse_loss(trajectory, full_trajectory, reduction="none").mean(dim = (1,2,3))
            best_idx = l2_loss.argmin(dim = 0)
            trajectory = trajectory[best_idx]

        return trajectory

    def embed_observation(self, obs_dict):

        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        print(B, To, value.shape)

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, To, Do
        cond = nobs_features.reshape(B, To, -1)

        return cond

    def embed_observation_dct(self, obs_dict):

        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        print(B, To, value.shape)

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        for key in nobs:
            x = nobs[key]
            x = x[:,:To,...].reshape(-1,*x.shape[2:])
            nobs[key] = x
        # this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))

        nobs_features = None
        # iterate over the images and pass them through the dct compressor
        camera_keys = ["agent_view", "wrist"]
        size = 0
        embedding = []
        print("here keys", nobs.keys())
        for key in nobs:
            if key in camera_keys:
                emb = compress_image_with_dct(nobs[key])
            else:
                emb = nobs[key]
                
            embedding.append(emb)
            size += emb.shape[1]
        
        B = emb.shape[0]
        n = 135 - size 
        embedding.append(torch.from_numpy(np.zeros((B, n))).cuda())
        # the rest are concatenated to the obs
        nobs_features = torch.hstack(embedding)
        # reshape back to B, To, Do
        cond = nobs_features.reshape(B, To, -1)

        return cond


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)

        if "embedding" in obs_dict:
            nobs["embedding"] = obs_dict["embedding"]

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        if self.hist_guidance.sampling.type == "fixed" or self.hist_guidance.sampling.type == "fixed_stabilization" or self.hist_guidance.sampling.type == "pyramid":
            assert len(self.hist_guidance.sampling.hist_len) == 1
            hl = self.hist_guidance.sampling.hist_len[0]
            hist_len = [hl if hl != "FULL" else self.n_obs_steps]
        elif self.hist_guidance.sampling.type == "average" or self.hist_guidance.sampling.type == "argmin":
            hist_len = []
            for l in self.hist_guidance.sampling.hist_len:
                if l == "FULL":
                    hist_len.append(self.n_obs_steps)
                else:
                    hist_len.append(l)
        elif self.hist_guidance.sampling.type == "full":
            hist_len = [self.n_obs_steps] # for dynamic we change it later
        elif self.hist_guidance.sampling.type == "dynamic":
                assert len(self.hist_guidance.sampling.hist_len) == 1
                hl = self.hist_guidance.sampling.hist_len[0]
                hist_len = [hl] * self.hist_guidance.sampling.n_samples + [self.n_obs_steps]
        else:
            raise ValueError(f"Invalid hist_guidance sampling type: {self.hist_guidance.sampling.type}")

        # handle different ways of passing observation
        if self.use_embed_if_present and "embedding" in obs_dict:
            cond = obs_dict["embedding"]
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
        shape = (B, T, Da)
        if self.pred_action_steps_only:
            shape = (B, self.n_action_steps, Da)

        obs_mask = torch.zeros((B, len(hist_len), To), device = device, dtype = torch.bool)
        cond = cond.unsqueeze(1).repeat(1, len(hist_len), 1, 1)
        for i, c in enumerate(hist_len):
            obs_mask[:,i,-c:] = True

        # run sampling
        nsample = self.conditional_sample(
            shape=shape,
            cond=cond,
            obs_mask = obs_mask,
            hist_len = hist_len,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch, debug=False):

        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        trajectory = nactions
        if self.use_embed_if_present and "embedding" in batch["obs"]:
            cond = batch["obs"]["embedding"]
        else:
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            cond = nobs_features.reshape(batch_size, To, -1)
        if self.pred_action_steps_only:
            start = To - 1
            end = start + self.n_action_steps
            trajectory = nactions[:,start:end]
        
        if self.hist_guidance.training.type == "random_independent":
            noise_levels = torch.rand((batch_size, To), device = self.device, dtype = torch.float32)
            cond = self._apply_noising(cond, noise_levels) 
        elif self.hist_guidance.training.type == "random_uniform":
            noise_levels = repeat(torch.rand((batch_size,), device = self.device, dtype = torch.float32), "b -> b t", t = hist_chunk_size)
            cond = self._apply_noising(cond, noise_levels) 
        elif self.hist_guidance.training.type == "binary_dropout":
            noise_levels = torch.bernoulli(0.5 * torch.ones((batch_size, To), device = self.device, dtype = torch.float32))
            cond = self._apply_noising(cond, noise_levels) 
        elif self.hist_guidance.training.type == "fixed" or self.hist_guidance.training.type == "full":
            # don't use noise levels when doing fixed training
            noise_levels = None
        else:
            raise ValueError(f"Invalid hist_guidance training type: {self.hist_guidance.training.type}")
        
        noise_levels = torch.cat([torch.zeros((batch_size, 1), device = self.device, dtype = torch.float32), noise_levels], dim = 1) # add a zero noise level for the timestep
        
        # scale noise levels to match for fourier embedding
        if not self.use_fourier_emb and noise_levels is not None:
            noise_levels = noise_levels * self.noise_scheduler.config.num_train_timesteps

        # train on multiple diffusion samples per obs, basically sampling
        # multiple noise levels.
        # NOTE: This increases the 'effective' batch-size, so batch_size != bsz
        if self.train_diffusion_n_samples != 1:
            def _repeat(x):
                return torch.repeat_interleave(
                    x, repeats=self.train_diffusion_n_samples, dim=0
                )

            cond = _repeat(cond)
            trajectory = _repeat(trajectory)
            if noise_levels is not None:
                noise_levels = _repeat(noise_levels)

        B_repeated = trajectory.shape[0] 

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        # input perturbation by adding additonal noise to alleviate exposure
        # bias reference: https://github.com/forever208/DDPM-IP
        if self.input_pertub != 0:
            noise = noise + self.input_pertub * torch.randn(
                trajectory.shape, device=trajectory.device
            )

        if self.use_flow_matching:
            # This is the original flow matching training objective with
            # standard Gaussian prior to approximate direction with model
            # output

            # Sample a random timestep for each image
            if self.fm_tsampler == "uniform":
                timestamps = torch.rand(
                    (B_repeated,), device=trajectory.device
                )
            elif self.fm_tsampler == "beta":
                timestamps = self.tsampler.sample((B_repeated,))
                timestamps = timestamps.to(trajectory.device)
            else:
                raise ValueError(f"Invalid tsampler: {self.fm_tsampler}")
            
            cont_t = timestamps.view(-1, *([1] * (noise.dim() - 1)))

            if self.use_fourier_emb:
                timesteps = timestamps.float()
            else:
                timesteps = (
                    timestamps * self.noise_scheduler.config.num_train_timesteps
                ).float()

            # Flow step: x0 -> x1
            x0, x1 = trajectory, noise
            direction = x1 - x0
            noisy_trajectory = x0 + cont_t * direction

            # for flow, the prediction type is always the velocity field
            pred = self.model(
                noisy_trajectory,
                timesteps,
                cond,
                noise_levels
            )
            target = direction
        else:
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B_repeated,), device=trajectory.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_trajectory = self.noise_scheduler.add_noise(
                trajectory, noise, timesteps)

            # Predict the noise residual
            pred = self.model(noisy_trajectory, timesteps, cond, noise_levels)

            pred_type = self.noise_scheduler.config.prediction_type 
            if pred_type == 'epsilon':
                target = noise
            elif pred_type == 'sample':
                target = trajectory
            else:
                raise ValueError(f"Unsupported prediction type {pred_type}")

        if not self.past_action_pred:
            pred = pred[:, self.n_obs_steps-1:]
            if debug:
                print(pred.shape[1], target.shape[1])
            target = target[:, self.n_obs_steps-1:]
        if self.past_steps_reg != -1:
            # print(self.n_obs_steps, self.past_steps_reg)
            assert self.n_obs_steps - self.past_steps_reg - 1 > 0
            pred = pred[:, self.n_obs_steps - self.past_steps_reg - 1:]
            # print(pred.shape)
            target = target[:, self.n_obs_steps - self.past_steps_reg - 1:]

        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
    
    def _apply_noising(self, nobs_features, noise_levels):
            """
            Applies noising to the observation features.
            """
            assert noise_levels.shape == nobs_features.shape[:2]

            noise = torch.randn_like(nobs_features)

            if self.use_flow_matching:
                direction = noise - nobs_features
                nobs_features = nobs_features + direction * noise_levels[:, :, None] # continuously "flow" towards the noise rather than adding noise
            else:
                noise_timesteps = (noise_levels * self.noise_scheduler.config.num_train_timesteps).long()[:, :, None] # reshape to (B, T, 1)
                alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=self.device)
                sqrt_alpha_prod = alphas_cumprod[noise_timesteps] ** 0.5
                sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[noise_timesteps]) ** 0.5
                nobs_features = sqrt_alpha_prod * nobs_features + sqrt_one_minus_alpha_prod * noise
            return nobs_features