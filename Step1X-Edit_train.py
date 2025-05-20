import argparse
import datetime
import json 
import itertools
import math
import os
import time
from pathlib import Path
from diffusers.optimization import get_scheduler
import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torchvision.transforms import functional as F
from tqdm import tqdm 
import copy
import sampling
from modules.autoencoder import AutoEncoder
from modules.conditioner import Qwen25VL_7b_Embedder as Qwen2VLEmbedder
from modules.model_edit import Step1XParams, Step1XEdit
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
)

def cudagc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def load_state_dict(model, ckpt_path, device="cuda", strict=False, assign=True):
    if Path(ckpt_path).suffix == ".safetensors":
        state_dict = load_file(ckpt_path, device)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(
        state_dict, strict=strict, assign=assign
    )
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    return model


def load_models(
    dit_path=None,
    ae_path=None,
    qwen2vl_model_path=None,
    device="cuda",
    max_length=256,
    dtype=torch.bfloat16,
):
    qwen2vl_encoder = Qwen2VLEmbedder(
        qwen2vl_model_path,
        device=device,
        max_length=max_length,
        dtype=dtype,
    )

    with torch.device("meta"):
        ae = AutoEncoder(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        )

        step1x_params = Step1XParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
        )
        dit = Step1XEdit(step1x_params)

    ae = load_state_dict(ae, ae_path, 'cpu')
    dit = load_state_dict(
        dit, dit_path, 'cpu'
    )

    ae = ae.to(dtype=torch.float32)

    return ae, dit, qwen2vl_encoder

def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

class ImageGenerator:
    def __init__(
        self,
        dit_path=None,
        ae_path=None,
        qwen2vl_model_path=None,
        device="cuda",
        max_length=640,
        dtype=torch.bfloat16,
        quantized=False,
        offload=False,
        args="",
        
    ) -> None:
        self.device = torch.device(device)
        self.ae, self.dit, self.llm_encoder = load_models(
            dit_path=dit_path,
            ae_path=ae_path,
            qwen2vl_model_path=qwen2vl_model_path,
            max_length=max_length,
            dtype=dtype,
        )
        self.ae.requires_grad_(False)
        self.llm_encoder.requires_grad_(False)
        # print("--------len(list(self.dit.parameters()))------------", len(list(self.dit.parameters())))
        for param in list(self.dit.parameters())[:500]:
            param.requires_grad = False
        if not quantized:
            self.dit = self.dit.to(dtype=torch.bfloat16)
        if not offload:
            self.dit = self.dit.to(device=self.device)
            self.ae = self.ae.to(device=self.device)
        
        self.quantized = quantized 
        self.offload = offload
        params_to_optimize = self.dit.parameters()
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=args.learning_rate, 
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            eps=args.adam_epsilon,
        )
    
        # Load scheduler and models
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(args.pretrained_model_name_or_path, 'scheduler'))
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

        self.params_before = {name: param.data.clone() for name, param in self.dit.named_parameters()}
    
    def prepare(self, prompt, img, ref_image, ref_image_raw):
        bs, _, h, w = img.shape
        bs, _, ref_h, ref_w = ref_image.shape

        assert h == ref_h and w == ref_w

        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)
        elif bs >= 1 and isinstance(prompt, str):
            prompt = [prompt] * bs

        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        ref_img = rearrange(ref_image, "b c (ref_h ph) (ref_w pw) -> b (ref_h ref_w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)

        img_ids = torch.zeros(h // 2, w // 2, 3)

        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        ref_img_ids = torch.zeros(ref_h // 2, ref_w // 2, 3)

        ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(ref_h // 2)[:, None]
        ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(ref_w // 2)[None, :]
        ref_img_ids = repeat(ref_img_ids, "ref_h ref_w c -> b (ref_h ref_w) c", b=bs)

        if isinstance(prompt, str):
            prompt = [prompt]
        if self.offload:
            self.llm_encoder = self.llm_encoder.to(self.device)
        txt, mask = self.llm_encoder(prompt, ref_image_raw)
        if self.offload:
            self.llm_encoder = self.llm_encoder.cpu()
            cudagc()

        txt_ids = torch.zeros(bs, txt.shape[1], 3)

        img = torch.cat([img, ref_img.to(device=img.device, dtype=img.dtype)], dim=-2)
        img_ids = torch.cat([img_ids, ref_img_ids], dim=-2)

        # Ensure all tensors are on the same device as img
        return {
            "img": img.to(self.device),
            "mask": mask.to(self.device),
            "img_ids": img_ids.to(self.device),
            "llm_embedding": txt.to(self.device),
            "txt_ids": txt_ids.to(self.device),
        }

    @staticmethod
    def process_diff_norm(diff_norm, k):
        pow_result = torch.pow(diff_norm, k)

        result = torch.where(
            diff_norm > 1.0,
            pow_result,
            torch.where(diff_norm < 1.0, torch.ones_like(diff_norm), diff_norm),
        )
        return result

    def denoise(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        llm_embedding: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: list[float],
        cfg_guidance: float = 4.5,
        mask=None,
        show_progress=False,
        timesteps_truncate=1.0,
    ):
        if self.offload:
            self.dit = self.dit.to(self.device)
            
        # Convert timesteps to the correct format
        if isinstance(timesteps, torch.Tensor):
            timesteps = timesteps.cpu().tolist()[0]
            
        t_vec = torch.full(
            (img.shape[0],), timesteps, dtype=img.dtype, device=self.device
        )
        
        # Move all inputs to the model's device
        txt, vec = self.dit.connector(
            llm_embedding.to(self.device), 
            t_vec.to(self.device), 
            mask.to(self.device) if mask is not None else None
        )
        
        pred = self.dit(
            img=img.to(self.device),
            img_ids=img_ids.to(self.device),
            txt=txt.to(self.device),
            txt_ids=txt_ids.to(self.device),
            y=vec.to(self.device),
            timesteps=t_vec.to(self.device),
        )
        
        if cfg_guidance != -1:
            cond, uncond = (
                pred[0 : pred.shape[0] // 2, :],
                pred[pred.shape[0] // 2 :, :],
            )
            pred = uncond + cfg_guidance * (cond - uncond)
            
        # if self.offload:
        #     self.dit = self.dit.cpu()
        #     cudagc()
            
        return pred[:, :img.shape[1] // 2].to(self.device)  # Ensure output is on the correct device

    @staticmethod
    def unpack(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

    @staticmethod
    def load_image(image):
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, Image.Image):
            image = F.to_tensor(image.convert("RGB"))
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, str):
            image = F.to_tensor(Image.open(image).convert("RGB"))
            image = image.unsqueeze(0)
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def output_process_image(self, resize_img, image_size):
        res_image = resize_img.resize(image_size)
        return res_image
    
    def input_process_image(self, img, img_size=512):
        # 1. 打开图片
        w, h = img.size
        r = w / h 

        if w > h:
            w_new = math.ceil(math.sqrt(img_size * img_size * r))
            h_new = math.ceil(w_new / r)
        else:
            h_new = math.ceil(math.sqrt(img_size * img_size / r))
            w_new = math.ceil(h_new * r)
        h_new = math.ceil(h_new) // 16 * 16
        w_new = math.ceil(w_new) // 16 * 16

        img_resized = img.resize((w_new, h_new))
        return img_resized, img.size

    @torch.inference_mode()
    def generate_image(
        self,
        prompt,
        negative_prompt,
        ref_images,
        num_steps,
        cfg_guidance,
        seed,
        num_samples=1,
        init_image=None,
        image2image_strength=0.0,
        show_progress=False,
        size_level=512,
    ):
        assert num_samples == 1, "num_samples > 1 is not supported yet."
        ref_images_raw, img_info = self.input_process_image(ref_images, img_size=size_level)
        width, height = ref_images_raw.width, ref_images_raw.height
        ref_images_raw = self.load_image(ref_images_raw)
        ref_images_raw = ref_images_raw.to(self.device)
        if self.offload:
            self.ae = self.ae.to(self.device)

        ref_images = self.ae.encode(ref_images_raw.to(self.device) * 2 - 1)
        if self.offload:
            self.ae = self.ae.cpu()
            cudagc()

        seed = int(seed)
        seed = torch.Generator(device="cpu").seed() if seed < 0 else seed

        t0 = time.perf_counter()

        if init_image is not None:
            init_image = self.load_image(init_image)
            init_image = init_image.to(self.device)
            init_image = torch.nn.functional.interpolate(init_image, (height, width))
            if self.offload:
                self.ae = self.ae.to(self.device)
            init_image = self.ae.encode(init_image.to() * 2 - 1)
            if self.offload:
                self.ae = self.ae.cpu()
                cudagc()
        
        x = torch.randn(
            num_samples,
            16,
            height // 8,
            width // 8,
            device=self.device,
            dtype=torch.bfloat16,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )

        timesteps = sampling.get_schedule(
            num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True
        )

        if init_image is not None:
            t_idx = int((1 - image2image_strength) * num_steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image.to(x.dtype)

        x = torch.cat([x, x], dim=0)
        ref_images = torch.cat([ref_images, ref_images], dim=0)
        ref_images_raw = torch.cat([ref_images_raw, ref_images_raw], dim=0)
        inputs = self.prepare([prompt, negative_prompt], x, ref_image=ref_images, ref_image_raw=ref_images_raw)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.denoise(
                **inputs,
                cfg_guidance=cfg_guidance,
                timesteps=timesteps,
                show_progress=show_progress,
                timesteps_truncate=1.0,
            )
            x = self.unpack(x.float(), height, width)
            if self.offload:
                self.ae = self.ae.to(self.device)
            x = self.ae.decode(x)
            if self.offload:
                self.ae = self.ae.cpu()
                cudagc()
            x = x.clamp(-1, 1)
            x = x.mul(0.5).add(0.5)

        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s.")
        images_list = []
        for img in x.float():
            images_list.append(self.output_process_image(F.to_pil_image(img), img_info))
        return images_list

    def train(
        self,
        prompt,
        negative_prompt,
        ref_images,
        obj_images,
        num_steps,
        global_step,
        cfg_guidance,
        seed,
        num_samples=1,
        init_image=None,
        image2image_strength=0.0,
        show_progress=False,
        size_level=512,
        args="",
    ):
        assert num_samples == 1, "num_samples > 1 is not supported yet."
        
        # Process and resize images
        ref_images_raw, img_info = self.input_process_image(ref_images, img_size=size_level)
        width, height = ref_images_raw.width, ref_images_raw.height
        ref_images_raw = self.load_image(ref_images_raw)
        ref_images_raw = ref_images_raw.to(self.device)  # Move to device

        obj_images_raw, img_info = self.input_process_image(obj_images, img_size=size_level)
        width, height = obj_images_raw.width, obj_images_raw.height
        obj_images_raw = self.load_image(obj_images_raw)
        obj_images_raw = obj_images_raw.to(self.device)  # Move to device
        
        # Encode images using AutoEncoder
        if self.offload:
            self.ae = self.ae.to(self.device)  # Move AE to device for encoding
        ref_images = self.ae.encode(ref_images_raw * 2 - 1)
        obj_images = self.ae.encode(obj_images_raw * 2 - 1)
        if self.offload:
            self.ae = self.ae.cpu()  # Move AE back to CPU after encoding
            cudagc()

        seed = int(seed)
        seed = torch.Generator(device="cpu").seed() if seed < 0 else seed

        t0 = time.perf_counter()

        # Handle init_image if provided
        if init_image is not None:
            init_image = self.load_image(init_image)
            init_image = init_image.to(self.device)
            init_image = torch.nn.functional.interpolate(init_image, (height, width))
            if self.offload:
                self.ae = self.ae.to(self.device)
            init_image = self.ae.encode(init_image.to() * 2 - 1)
            if self.offload:
                self.ae = self.ae.cpu()
                cudagc()

        # Sample noise on the correct device
        noise = torch.randn(
            num_samples,
            16,
            height // 8,
            width // 8,
            device=self.device,
            dtype=torch.bfloat16,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )

        # Sample a random timestep for each image
        bsz = ref_images.shape[0]
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=num_samples,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=ref_images.device)
        
        # Define get_sigmas function locally with correct device handling
        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = self.noise_scheduler_copy.sigmas.to(device=self.device, dtype=dtype)
            schedule_timesteps = self.noise_scheduler_copy.timesteps.to(self.device)
            timesteps = timesteps.to(self.device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    
            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma
            
        sigmas = get_sigmas(timesteps, n_dim=ref_images.ndim, dtype=ref_images.dtype)
        noisy_model_input = (1.0 - sigmas) * ref_images + sigmas * noise

        # Prepare inputs for denoising
        x = torch.cat([noisy_model_input, noisy_model_input], dim=0).to(self.device)
        ref_images = torch.cat([ref_images, obj_images], dim=0).to(self.device)
        ref_images_raw = torch.cat([ref_images_raw, obj_images_raw], dim=0).to(self.device)
        inputs = self.prepare([prompt, negative_prompt], x, ref_image=ref_images, ref_image_raw=ref_images_raw)

        # Denoise with proper device management
        # if self.offload:
        #     self.dit = self.dit.to(self.device)  # Move DIT to device for computation
            
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.denoise(
                **inputs,
                cfg_guidance=cfg_guidance,
                timesteps=timesteps,
                show_progress=show_progress,
                timesteps_truncate=1.0,
            )
            x = self.unpack(x.float(), height, width)
            


        latents = obj_images
        model_pred = x
        
        # Compute loss
        if args.precondition_outputs:
            target = latents
        else:
            target = noise - latents

        loss = torch.mean(
            ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        print("loss:", loss)
        
        # Ensure loss is on the correct device before backpropagation
        loss = loss.to(self.device)


        # 在关键位置添加调试输出
        print(f"Model device: {next(self.dit.parameters()).device}")
        print(f"Input tensor device: {ref_images.device}")
        print(f"Loss device: {loss.device}")
        loss.backward()
        
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        if self.offload:
            self.dit = self.dit.cpu()  # Move DIT back to CPU after computation
            cudagc()
        if global_step!=0 and global_step % args.save_steps == 0:
            """检查并打印被更新的参数"""
            updated_params = []
            current_state = self.dit.state_dict()
            for name in self.params_before:
                # 将当前参数移到与保存的参数相同的设备
                current_param = current_state[name].to(self.params_before[name].device)
                if not torch.allclose(self.params_before[name], current_param, atol=1e-6):
                    updated_params.append(name)
            
            if updated_params:
                print(f"\n===== 检测到 {len(updated_params)} 个参数更新 =====")
                for param_name in updated_params:
                    print(f"参数 {param_name} 已更新")
            else:
                print("\n===== 未检测到参数更新 =====")



            save_path = os.path.join(args.output_dir, str(global_step))
            print("save_path:", save_path)
            # Ensure the folder exists
            if not os.path.exists(save_path):
                os.makedirs(save_path)
    
    
            import safetensors.torch
            model_path = os.path.join(save_path, 'pytorch_model.safetensors')
            safetensors.torch.save_file(self.dit.state_dict(), model_path)
            print("save model", global_step)






def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,  default='./models',  help='Path to the model checkpoint')
    parser.add_argument('--input_dir', type=str, default='./dataset', help='Path to the input image directory')
    parser.add_argument('--json_path', type=str, default='./examples/prompt_en.json', help='Path to the JSON file containing image names and prompts')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')
    parser.add_argument('--num_steps', type=int, default=28, help='Number of diffusion steps')
    parser.add_argument('--cfg_guidance', type=float, default=6.0, help='CFG guidance strength')
    parser.add_argument('--size_level', default=128, type=int)
    parser.add_argument('--offload', action='store_true', default=True, help='Use offload for large models')
    parser.add_argument('--quantized', action='store_true', default=False,help='Use fp8 model weights')
    parser.add_argument(
    "--output_dir",
    type=str,
    default="my_output",
    help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
    "--save_steps",
    type=int,
    default=50,
    help=(
        "Save a checkpoint of the training state every X updates"
    ),)
    parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-1,
    help="Learning rate to use.",)
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )  
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument(
    "--adam_epsilon",
    type=float,
    default=1e-08,
    help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
    "--max_train_steps",
    type=int,
    default=None,
    help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path_scheduler",
        type=str,
        default='FlowMatchEulerDiscreteScheduler path',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
    "--precondition_outputs",
    type=int,
    default=1,
    help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
    "model `target` is calculated.",
    )
    args = parser.parse_args()

    assert os.path.exists(args.input_dir), f"Input directory {args.input_dir} does not exist."
    assert os.path.exists(args.json_path), f"JSON file {args.json_path} does not exist."

    args.output_dir = args.output_dir.rstrip('/') + ('-offload' if args.offload else "") + ('-quantized' if args.quantized else "") + f"-{args.size_level}"
    os.makedirs(args.output_dir, exist_ok=True)

    image_and_prompts = json.load(open(args.json_path, 'r'))

    image_edit = ImageGenerator(
        ae_path=os.path.join(args.model_path, 'vae.safetensors'),
        dit_path=os.path.join(args.model_path, "step1x-edit-i1258-FP8.safetensors" if args.quantized else "step1x-edit-i1258.safetensors"),
        qwen2vl_model_path=('Qwen/Qwen2.5-VL-7B-Instruct'),
        max_length=640,
        quantized=args.quantized,
        offload=args.offload,
        args=args,
    )

    global_step = 0
    all_step = 10000
    for global_step in range(all_step):
        prompt = "Replace the spoon with a fork."
        input_image_name = "demo-input.png"
        output_image_name = "demo-output.png"
        print('train global_step:', global_step)
        input_image_path = os.path.join(args.input_dir, input_image_name)
        output_image_path = os.path.join(args.input_dir, output_image_name)
        image_edit.train(
                    prompt,
                    negative_prompt="",
                    ref_images=Image.open(input_image_path).convert("RGB"),
                    obj_images=Image.open(output_image_path).convert("RGB"),
                    num_samples=1,
                    num_steps=args.num_steps,
                    global_step=global_step,
                    cfg_guidance=args.cfg_guidance,
                    seed=args.seed,
                    show_progress=True,
                    size_level=args.size_level,
                    args=args,
                )
        global_step+=1

if __name__ == "__main__":
    main()
