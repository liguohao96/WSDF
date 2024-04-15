import torch
import torch.nn as nn

import numpy as np

from contextlib import nullcontext

torch_compile_fn = getattr(torch, "compile", lambda f, **kwargs:f)

with nullcontext("diffusion"):
    from transformers import CLIPTextModel, CLIPTokenizer, logging
    from diffusers import StableDiffusionPipeline
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from diffusers.utils.import_utils import is_xformers_available
    from diffusers import (
        DDPMScheduler,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        DiffusionPipeline
    )
    from diffusers.models.vae import DiagonalGaussianDistribution
    from diffusers.utils.import_utils import is_torch_version

    class StableDiffusion(nn.Module):
        def __init__(self, 
                     device, 
                     sd_version = '2.1',
                     hf_key = None,
                     guidance_weight = 7.5, 
                     guidance_method = "SDS",
                     time_step_range = [0.02, 0.5],
                     release_pipeline    = True,
                     release_decoder     = True,
                     release_textencoder = True,
                     gradient_checkpoint = False,
                     ):
            super().__init__()

            self.device = device
            self.sd_version = sd_version

            self.guidance_method = guidance_method
            print(f'[INFO] loading stable diffusion... {sd_version}')
            if hf_key is not None:
                print(f"[INFO] using hugging face custom model key: {hf_key}")
                model_key = hf_key
            else:
                if self.sd_version == '2.1':
                    model_key = "stabilityai/stable-diffusion-2-1-base"
                elif self.sd_version == '2.0':
                    model_key = "stabilityai/stable-diffusion-2-base"
                elif self.sd_version == '1.5':
                    model_key = "runwayml/stable-diffusion-v1-5"
            
            self.model_key = model_key

            pipe = StableDiffusionPipeline.from_pretrained(
                model_key,
                # torch_dtype=torch.float16,
                # use_safetensors=True,
            )

            #################### VRAM usage optimization #######################
            pipe.enable_vae_slicing()
            # pipe.enable_model_cpu_offload()

            import diffusers.models.attention_processor
            is_pytorch2 = int(torch.__version__.split('.')[0]) == 2
            if is_pytorch2 and hasattr(diffusers.models.attention_processor, "AttnProcessor2_0"):
                from diffusers.models.attention_processor import AttnProcessor2_0
                print(f"using torch2.0 attension")
                pipe.unet.set_attn_processor(AttnProcessor2_0())
                pipe.vae.set_attn_processor(AttnProcessor2_0())
            elif is_xformers_available():
                print(f"using xformer attension")
                pipe.unet.enable_xformers_memory_efficient_attention()
            else:
                print(f"torch version: {torch.__version__}")
                print("!!! StableDiffusion is no accelerated !!!")

            # this will double training time and batch size !
            if pipe.vae._supports_gradient_checkpointing and gradient_checkpoint:
                pipe.vae.enable_gradient_checkpointing()
            print("gradient checkpoint", pipe.vae.is_gradient_checkpointing)

            self.vae          = pipe.vae.to(device, torch.float32)
            self.tokenizer    = pipe.tokenizer
            self.text_encoder = pipe.text_encoder.to(device, torch.float16)
            self.unet         = pipe.unet.to(device, torch.float16)

            self.unet         = torch_compile_fn(self.unet, mode="reduce-overhead", fullgraph=True)
            # self.vae.encoder = torch.compile(self.vae.encoder, mode="reduce-overhead", fullgraph=True)

            if release_pipeline:
                del pipe
            if release_decoder:
                del self.vae.decoder
            if release_textencoder:
                del self.text_encoder

            self.SDS_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=torch.float16)

            if not hasattr(self.SDS_scheduler, "sigmas"):
                self.SDS_scheduler.sigmas = ((1 - self.SDS_scheduler.alphas_cumprod) / self.SDS_scheduler.alphas_cumprod) ** 0.5

            self.noise_gen = torch.Generator(self.device)
            self.noise_gen.manual_seed(0)

            # self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience


            self.num_train_timesteps = self.SDS_scheduler.config.num_train_timesteps
            self.min_step = int(self.num_train_timesteps *  time_step_range[0])
            self.max_step = int(self.num_train_timesteps *  time_step_range[1])
            # self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
            self.guidance_weight = guidance_weight
            print(f'[INFO] loaded stable diffusion!')

        def get_text_embeds(self, prompt, batch=1):
            text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
            with torch.no_grad():
                device = next(self.text_encoder.parameters()).device
                print("text_encoder.device", device)
                text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
            if batch > 1:
                text_embeddings = text_embeddings.repeat(batch, 1, 1)

            return text_embeddings
    
        def get_uncond_embeds(self, negative_prompt, batch):
            uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
            with torch.no_grad():
                uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
    
            if batch > 1:
                uncond_embeddings = uncond_embeddings.repeat(batch, 1, 1)
            return uncond_embeddings

        def encode_imgs(self, imgs):
            # imgs: [B, 3, H, W]
            # if self.mode == 'appearance_modeling':

            imgs = 2 * imgs - 1

            device = next(self.vae.parameters()).device
            # posterior = self.vae.encode(imgs.to(device)).latent_dist

            from contextlib import nullcontext
            with nullcontext("AutoEncoderKL.encode"):
                # h = self.vae.encoder(imgs)
                with nullcontext("Encoder.forward"):
                    sample = imgs
                    sample = self.vae.encoder.conv_in(sample)
                    if self.vae.encoder.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward

                        # down
                        if is_torch_version(">=", "1.11.0"):
                            for down_block in self.vae.encoder.down_blocks:
                                sample = torch.utils.checkpoint.checkpoint(
                                    create_custom_forward(down_block), sample, use_reentrant=False
                                )
                            # middle
                            sample = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(self.vae.encoder.mid_block), sample, use_reentrant=False
                            )
                        else:
                            for down_block in self.vae.encoder.down_blocks:
                                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                            # middle
                            sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

                    else:
                        # down
                        for down_block in self.vae.encoder.down_blocks:
                            sample = down_block(sample)

                        # middle
                        sample = self.vae.encoder.mid_block(sample)

                    # post-process
                    sample = self.vae.encoder.conv_norm_out(sample)
                    sample = self.vae.encoder.conv_act(sample)
                    sample = self.vae.encoder.conv_out(sample)

                    h = sample

                moments = self.vae.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)

            latents = posterior.sample() * self.vae.config.scaling_factor

            return latents.to(imgs.device)

        def decode_latents(self, latents):

            latents = 1 / self.vae.config.scaling_factor * latents

            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)

            return imgs

        def guidance(self, *args, **kwargs):
            if self.guidance_method == "SDS":
                return self.SDS_guidance(*args, **kwargs)
            else:
                raise NotImplementedError

        def SDS_guidance(self, latents, t, uncond_text_z, cond_text_z, cond_text_weight=None, uncond_image_z=None, cond_image_z=None,
                         noise=None, add_noise=True, scheduler=None, cfg_weight=None):
            BS = latents.size(0)

            CFG_WEIGHT = self.guidance_weight if cfg_weight is None else cfg_weight

            scheduler = self.SDS_scheduler if scheduler is None else scheduler

            with torch.no_grad():
                text_embeddings = torch.cat([uncond_text_z.unsqueeze(0), cond_text_z], dim=0) # [Unc,Pos,Neg_0,Neg_1], B
                N_EXPAND        = text_embeddings.size(0)
                text_embeddings = text_embeddings.flatten(0, 1)
                    
                if uncond_image_z is not None:
                    image_embeds      = torch.cat([guidance.img_uc, guidance.img_c])
                    added_cond_kwargs = {"image_embeds": image_embeds}
                else:
                    added_cond_kwargs = {}

                # add noise
                noise = torch.randn_like(latents)              if noise is None else noise
                codes = scheduler.add_noise(latents, noise, t) if add_noise else latents

                input = codes.unsqueeze(0).expand(N_EXPAND, -1, -1, -1, -1).flatten(0, 1)    # K*B, ...
                tt    = t.reshape(1, -1).expand(N_EXPAND, -1).flatten(0, 1).to(input.device) # k*B
                input = scheduler.scale_model_input(input, tt[0])
                noise_pred = self.unet(input, tt, encoder_hidden_states=text_embeddings,
                                    added_cond_kwargs=added_cond_kwargs).sample

                noise_pred_uncond, noise_pred_text = torch.tensor_split(noise_pred, (BS,), dim=0)
                if noise_pred_text.size(0) == noise_pred_uncond.size(0):
                    # CFG
                    noise_pred = noise_pred_uncond + CFG_WEIGHT * (noise_pred_text - noise_pred_uncond) # [B, 4, 64, 64]
                else:
                    # Perp-Neg
                    delta_noise_preds = noise_pred_text - noise_pred_uncond[None].expand(N_EXPAND-1,-1,-1,-1,-1).flatten(0,1)
                    weights   = cond_text_weight.flatten()
                    delta_DSD = PerpNegUtil.weighted_perpendicular_aggregator(delta_noise_preds,\
                                                                    weights, BS)

                    noise_pred = noise_pred_uncond + CFG_WEIGHT * delta_DSD

                alphas = scheduler.alphas_cumprod
                w = alphas[t] ** 0.5 * (1 - alphas[t])
                w = w[:, None, None, None] # [B, 1, 1, 1]
                return noise_pred, noise, w
        
with nullcontext("PerpNeg"):
    class PerpNegUtil:
        # Please refer to the https://perp-neg.github.io/ for details about the paper and algorithm
        def get_perpendicular_component(x, y):
            assert x.shape == y.shape
            return x - ((torch.mul(x, y).sum())/max(torch.norm(y)**2, 1e-6)) * y


        def batch_get_perpendicular_component(x, y):
            assert x.shape == y.shape
            result = []
            for i in range(x.shape[0]):
                result.append(PerpNegUtil.get_perpendicular_component(x[i], y[i]))
            return torch.stack(result)


        def weighted_perpendicular_aggregator(delta_noise_preds, weights, batch_size):
            """ 
            Notes: 
             - weights: an array with the weights for combining the noise predictions
             - delta_noise_preds: [B x K, 4, 64, 64], K = max_prompts_per_dir
            """
            delta_noise_preds = delta_noise_preds.split(batch_size, dim=0) # K x [B, 4, 64, 64]
            weights = weights.split(batch_size, dim=0) # K x [B]
            # print(f"{weights[0].shape = } {weights = }")

            assert torch.all(weights[0] == 1.0)

            main_positive = delta_noise_preds[0] # [B, 4, 64, 64]

            accumulated_output = torch.zeros_like(main_positive)
            for i, complementary_noise_pred in enumerate(delta_noise_preds[1:], start=1):
                # print(f"\n{i = }, {weights[i] = }, {weights[i].shape = }\n")
                weight_i = weights[i]
                idx_non_zero = torch.abs(weight_i) > 1e-4

                # print(f"{idx_non_zero.shape = }, {idx_non_zero = }")
                # print(f"{weights[i][idx_non_zero].shape = }, {weights[i][idx_non_zero] = }")
                # print(f"{complementary_noise_pred.shape = }, {complementary_noise_pred[idx_non_zero].shape = }")
                # print(f"{main_positive.shape = }, {main_positive[idx_non_zero].shape = }")
                if sum(idx_non_zero) == 0:
                    continue
                accumulated_output[idx_non_zero] += weight_i[idx_non_zero].reshape(-1, 1, 1, 1) * PerpNegUtil.batch_get_perpendicular_component(complementary_noise_pred[idx_non_zero], main_positive[idx_non_zero])

            #assert accumulated_output.shape == main_positive.shape,# f"{accumulated_output.shape = }, {main_positive.shape = }"


            return accumulated_output + main_positive

        @staticmethod
        def get_pos_neg_text_embeddings(embeddings, azimuth_val, front_decay_factor=1, side_decay_factor=1, negative_w=1):
            if azimuth_val >= -90 and azimuth_val < 90:
                if azimuth_val >= 0:
                    r = 1 - azimuth_val / 90
                else:
                    r = 1 + azimuth_val / 90
                start_z = embeddings['front']
                end_z   = embeddings['side']
                # if random.random() < 0.3:
                #     r = r + random.gauss(0, 0.08)
                pos_z = r * start_z + (1 - r) * end_z
                text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
                if r > 0.8:
                    front_neg_w = 0.0
                else:
                    front_neg_w = np.exp(-r * front_decay_factor) * negative_w
                if r < 0.2:
                    side_neg_w = 0.0
                else:
                    side_neg_w = np.exp(-(1-r) * side_decay_factor) * negative_w

                weights = torch.tensor([1.0, front_neg_w, side_neg_w])
            else:
                if azimuth_val >= 0:
                    r = 1 - (azimuth_val - 90) / 90
                else:
                    r = 1 + (azimuth_val + 90) / 90
                start_z = embeddings['side']
                end_z   = embeddings['back']
                # if random.random() < 0.3:
                #     r = r + random.gauss(0, 0.08)
                pos_z = r * start_z + (1 - r) * end_z
                text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
                front_neg_w = negative_w 
                if r > 0.8:
                    side_neg_w = 0.0
                else:
                    side_neg_w = np.exp(-r * side_decay_factor) * negative_w / 2

                weights = torch.tensor([1.0, side_neg_w, front_neg_w])
            return text_z, weights.to(text_z.device, non_blocking=True)
