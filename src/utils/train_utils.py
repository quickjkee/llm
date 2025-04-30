import copy
from typing import Optional

import torch
import numpy as np
import random
import pandas as pd
import torch.distributed as dist

from contextlib import nullcontext
from tqdm import tqdm
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from transformers import (
    CLIPTextModelWithProjection,
    T5EncoderModel,
)
from torchvision.transforms import ToPILImage

if is_wandb_available():
    import wandb


########################################################################################################################
#                                       UTILS FUNCTIONS FOR TRAIN                                                      #
########################################################################################################################


VALIDATION_PROMPTS = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
]

# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def log_validation(
        args,
        accelerator,
        tokenizer, text_embedding_layer_llm, transformer_llm,
        fm_solver, noise_scheduler,
        logger, global_step, image_processor, vae,
):
    # Set validation prompts
    if args.validation_prompt is not None:
        logger.info(
            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
            f" {args.validation_prompt}."
        )
        validation_prompts = [args.validation_prompt]
    else:
        validation_prompts = VALIDATION_PROMPTS

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    weight_dtype = torch.float16

    image_logs = []
    for _, prompt in enumerate(validation_prompts):
        # Sample batch in a loop to save memory
        inputs = tokenizer([prompt],
                           padding=True,
                           return_tensors="pt").to(accelerator.device)
        embeds_llm = text_embedding_layer_llm(inputs["input_ids"])
        mask_llm = inputs["attention_mask"]

        noise_scheduler_ = copy.deepcopy(noise_scheduler)
        noise_scheduler_.set_timesteps(28)
        sigmas = noise_scheduler_.sigmas
        idx_start = torch.tensor([0] * len(embeds_llm))
        idx_end = torch.tensor([len(sigmas) - 1] * len(embeds_llm))
        sampling_fn = fm_solver.flow_matching_sampling

        images = []
        for _ in range(1):
            latent = torch.randn(
                (1, 16, 128, 128), 
                generator=generator, 
                device=accelerator.device
            )
            image = sampling_fn(
                transformer_llm,
                latent,
                embeds_llm, 
                mask_llm,
                idx_start, idx_end,
                sigmas=sigmas,
            ).to(weight_dtype)
            
            latent = (image / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(latent, return_dict=False)[0]
            image = image_processor.postprocess(image, output_type='pil')[0]
            images.append(image)

        image_logs.append({"validation_prompt": prompt, "images": images})
    
    torch.cuda.empty_cache()
    k = 0       
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image.resize((512, 512))))
                    #image.save(f'{args.output_dir}/{global_step}_{k}.jpg')
                    #k += 1

                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    return images
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def distributed_sampling(
    pipeline,
    args,
    val_prompt_path,
    prepare_prompt_embed_from_caption,
    solver,
    noise_scheduler,
    accelerator,
    logger,
    offloadable_encoders=None,
    cfg_scale=0.0,
):
    logger.info(f"Running sampling")

    weight_dtype = pipeline.text_encoder.dtype
    assert weight_dtype == torch.float16
    
    offloadable_encoders = offloadable_encoders or []
    
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    # Prepare validation prompts
    rank_batches, rank_batches_index, all_prompts = prepare_val_prompts(
        val_prompt_path, bs=args.eval_batch_size, max_cnt=args.max_eval_samples
    )

    local_images = []
    local_text_idxs = []
    
    # Load text encoders to device
    for encoder in offloadable_encoders:
        encoder.to(accelerator.device)
    torch.cuda.empty_cache()
                
    for cnt, mini_batch in enumerate(tqdm(rank_batches, disable=(not accelerator.is_main_process))):        
        prompt_embeds, pooled_prompt_embeds = prepare_prompt_embed_from_caption(
            list(mini_batch), pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3,
            pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3
        )

        if cfg_scale > 1.0:
            uncond_prompt_embeds, uncond_pooled_prompt_embeds = prepare_prompt_embed_from_caption(
                [' '] * len(prompt_embeds),
                pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3,
                pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3
            )
        else:
            uncond_prompt_embeds, uncond_pooled_prompt_embeds = None, None
                
        sigmas = noise_scheduler.sigmas[solver.boundary_idx]
        timesteps = noise_scheduler.timesteps[solver.boundary_start_idx]
        idx_start = torch.tensor([0] * len(prompt_embeds))
        idx_end = torch.tensor([len(solver.boundary_idx) - 1] * len(prompt_embeds))

        sampling_fn = solver.flow_matching_sampling_stochastic if args.stochastic_case else solver.flow_matching_sampling
        latent = torch.randn(
            (len(prompt_embeds), 16, 128, 128),
            generator=generator, device=accelerator.device
        )
        images = sampling_fn(
            pipeline.transformer, latent,
            prompt_embeds, pooled_prompt_embeds,
            uncond_prompt_embeds, uncond_pooled_prompt_embeds,
            idx_start, idx_end,
            cfg_scale=cfg_scale, do_scales=True if args.scales else False,
            sigmas=sigmas, timesteps=timesteps, generator=generator
        ).to(weight_dtype)

        latent = (images / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        images = pipeline.vae.decode(latent, return_dict=False)[0]
        images = pipeline.image_processor.postprocess(images, output_type='pil')

        for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
            img_tensor = torch.tensor(np.array(images[text_idx].resize((512, 512))))
            local_images.append(img_tensor)
            local_text_idxs.append(global_idx)

    # Offload text encoders back
    for encoder in offloadable_encoders:
        encoder.cpu()
    torch.cuda.empty_cache()

    local_images = torch.stack(local_images).cuda()
    local_text_idxs = torch.tensor(local_text_idxs).cuda()

    gathered_images = accelerator.gather(local_images).cpu().numpy()
    gathered_text_idxs = accelerator.gather(local_text_idxs).cpu().numpy()

    images, prompts = [], []
    if accelerator.is_main_process:
        for image, global_idx in zip(gathered_images, gathered_text_idxs):
            images.append(ToPILImage()(image))
            prompts.append(all_prompts[global_idx])

    accelerator.wait_for_everyone()
    return images, prompts
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_val_prompts(path, bs=20, max_cnt=5000):
    df = pd.read_csv(path)
    all_text = list(df['caption'])
    all_text = all_text[:max_cnt]

    num_batches = ((len(all_text) - 1) // (bs * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]
    return rank_batches, rank_batches_index, all_text
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model
# ----------------------------------------------------------------------------------------------------------------------

