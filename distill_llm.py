#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion XL for text2image with support for LoRA."""
import copy
import logging
import math
import os
from copy import deepcopy
from pathlib import Path

import random
import datasets
import torch
import torch.utils.checkpoint
import transformers
import numpy as np
import diffusers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict, get_peft_model
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.loaders import SD3LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers

from yt_tools.nirvana_utils import copy_snapshot_to_out, copy_out_to_snapshot, copy_logs_to_logs_path
from src.data import create_dataloader
from src.utils.train_utils import log_validation, unwrap_model,distributed_sampling
from src.utils.flow_matching_sampler import FlowMatchingSolver
from src.pipelines.stable_diffusion_3 import ScaleWiseStableDiffusion3Pipeline
from src.models.transformer_llm_dm import TransformerLLM
from src.utils.distillation_losses import diffusion_loss
from src.utils.metrics import calculate_scores

logger = get_logger(__name__)

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}

########################################################################################################################
#                                               TRAINING                                                               #
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
def train(args):

    ## PREPARATION STAGE
    ## -----------------------------------------------------------------------------------------------
    ## Prepare accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = prepare_accelertor(args, logging_dir, True)

    ## Some useful asserts
    assert args.max_eval_samples % accelerator.num_processes == 0, \
        "Must be divisible by world size. Otherwise, allgather fails."

    ## Torch setup for SD3
    torch.set_float32_matmul_precision('high')
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True

    ## Prepare models
    transformer_llm, transformer_dm, vae, \
    tokenizer, text_embedding_layer_llm, \
    noise_scheduler, weight_dtype = prepare_models(args, accelerator)

    ## LLM+DM
    transformer_llm = TransformerLLM(transformer_llm,
                                     transformer_dm.pos_embed,
                                     transformer_dm.inner_dim)
    cast_training_params(transformer_llm, dtype=torch.float32)
    transformer_llm = accelerator.prepare(transformer_llm)
    transformer_dm = accelerator.prepare(transformer_dm)
    text_embedding_layer_llm = accelerator.prepare(text_embedding_layer_llm)

    ## Prepare optimizers
    optimizer, lr_scheduler, params_to_optimize = prepare_optimizer(args, transformer_llm)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    ## Prepare data
    train_dataloader = create_dataloader(args.train_dataloader_config_path, args.train_batch_size)
    train_dataset = train_dataloader.dataset

    ## Load if exist
    initial_global_step = load_if_exist(args, accelerator, len(train_dataloader))

    ## Prepare 3rd party utils
    prepare_3rd_party(args, accelerator, len(train_dataloader))
    ## -----------------------------------------------------------------------------------------------

    ## TRAINING STAGE
    ## -----------------------------------------------------------------------------------------------
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    if hasattr(train_dataset, "__len__"):
        logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    assert transformer_llm.training
    for batch in train_dataloader:

        ### Sample a batch and calculate the loss
        ### ----------------------------------------------------
        latent_image, embeds_llm, mask_llm, prompt_embeds_dm, pooled_prompt_embeds_dm = sample_batch(
            args, accelerator, batch,
            tokenizer, text_embedding_layer_llm,
            vae, weight_dtype
        )

        u = torch.rand(size=(latent_image.shape[0],))
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(accelerator.device)
        noise = torch.randn_like(latent_image)
        noisy_latent_image = noise_scheduler.scale_noise(latent_image, timesteps, noise)

        avg_dm_loss = diffusion_loss(
            transformer_llm, transformer_dm,
            prompt_embeds_dm, pooled_prompt_embeds_dm, embeds_llm, mask_llm,
            noisy_latent_image, timesteps,
            optimizer, lr_scheduler, params_to_optimize,
            accelerator, args,
        )
        ### ----------------------------------------------------

        progress_bar.update(1)
        global_step += 1

        ### Log validation images
        ### ----------------------------------------------------
        if accelerator.is_main_process:
            if global_step % args.validation_steps == 0:
                image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)
                fm_solver = FlowMatchingSolver(noise_scheduler)

                log_validation(
                    args, accelerator,
                    tokenizer, 
                    unwrap_model(text_embedding_layer_llm, accelerator), 
                    unwrap_model(transformer_llm, accelerator),
                    fm_solver, noise_scheduler,
                    logger, global_step, image_processor, vae
                )
                copy_out_to_snapshot(args.output_dir)
                copy_logs_to_logs_path(logging_dir)

                torch.cuda.empty_cache()

        accelerator.wait_for_everyone()
        ### ----------------------------------------------------

        logs = {
            "dm_loss": avg_dm_loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break
    ## -----------------------------------------------------------------------------------------------

    accelerator.wait_for_everyone()
    accelerator.end_training()
# ----------------------------------------------------------------------------------------------------------------------


########################################################################################################################
#                                           TRAINING HELPER FUNCTIONS                                                  #
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
def sample_batch(args, accelerator, batch,
                 tokenizer, llm_embedding,
                 vae, weight_dtype):
    device = 'cuda'
    pixel_values_ = batch[args.image_column].to(device=accelerator.device)

    # Memory efficient VAE encoding
    model_inputs = []
    for pixel_value in pixel_values_:
        model_input = vae.encode(pixel_value[None].to(weight_dtype)).latent_dist.sample()
        model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
        model_inputs.append(model_input)
    latent_image = torch.cat(model_inputs, dim=0)
    assert latent_image.dtype == weight_dtype

    # Text embeddings for DM
    prompt_embeds, pooled_prompt_embeds = batch.get(args.text_embedding_column).to(device), \
                                          batch.get(args.pooled_text_embedding_column).to(device)
    prompt_embeds_2, pooled_prompt_embeds_2 = batch.get(args.text_embedding_2_column).to(device), \
                                              batch.get(args.pooled_text_embedding_2_column).to(device)
    t5_prompt_embeds = batch.get(args.text_embedding_3_column).to(device)

    clip_prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds_dm = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
    pooled_prompt_embeds_dm = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds_2], dim=-1)

    # Text embeddings for LLM
    captions = []
    for caption in batch[args.text_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if True else caption[0])
        else:
            raise ValueError(
                f"Caption column `{args.text_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(captions,
                      padding=True,
                      return_tensors="pt").to(accelerator.device)
    embeds_llm = llm_embedding(inputs["input_ids"])
    mask_llm = inputs["attention_mask"]

    return latent_image, embeds_llm, mask_llm, prompt_embeds_dm, pooled_prompt_embeds_dm
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_accelertor(args, logging_dir, find_unused_parameters=False):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    return accelerator
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_models(args, accelerator):

    # Diffusion specific
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path_dm, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path_dm, subfolder="vae", revision=args.revision, variant=args.variant
    )
    transformer_dm = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path_dm, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    # LLM specific
    transformer_llm = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name_or_path_llm,
                                                           output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path_llm)
    tokenizer.pad_token = tokenizer.eos_token
    text_embedding_layer_llm = copy.deepcopy(transformer_llm.model.embed_tokens)

    # We only train the additional adapter LoRA layers
    transformer_dm.requires_grad_(False)
    transformer_llm.requires_grad_(False)
    text_embedding_layer_llm.requires_grad_(False)
    vae.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move transformer, vae and text_encoder to device and cast to weight_dtype
    transformer_dm.to(accelerator.device, dtype=weight_dtype, memory_format=torch.channels_last)
    transformer_llm.to(accelerator.device, dtype=weight_dtype)
    text_embedding_layer_llm.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype, memory_format=torch.channels_last)

    # add new LoRA weights
    # Set correct lora layers
    # Default choice
    target_modules = []
    if args.apply_lora_to_attn_projections:
        target_modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
    assert len(target_modules) > 0, "LoRA has to be applied to at least one type of projection."

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer_llm = get_peft_model(transformer_llm, transformer_lora_config)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the transformer attn processor layers
            # or there are the transformer and text encoder attn layers
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model, accelerator), type(unwrap_model(transformer, accelerator))):
                    transformer_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            ScaleWiseStableDiffusion3Pipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer_llm, accelerator))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = SD3LoraLoaderMixin.lora_state_dict(input_dir)
        transformer_state_dict = lora_state_dict
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if weight_dtype == torch.float16:
            models = [transformer_]
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer_llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False},)
        transformer_dm.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if weight_dtype == torch.float16:
        models = [transformer_llm]
        cast_training_params(models, dtype=torch.float32)

    return transformer_llm, transformer_dm, vae, \
           tokenizer, text_embedding_layer_llm, \
           noise_scheduler, weight_dtype
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_prompt_embed_from_caption(
        caption, tokenizer, tokenizer_2, tokenizer_3,
        text_encoder, text_encoder_2, text_encoder_3
):
    uncond_tokens = {"input_ids": tokenize_prompt(tokenizer, caption),
                     "input_ids_2": tokenize_prompt(tokenizer_2, caption),
                     "input_ids_3": tokenize_prompt(tokenizer_3, caption),
                     }
    uncond_prompt_embeds, uncond_pooled_prompt_embeds = encode_prompt(
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3,
        input_ids=uncond_tokens["input_ids"],
        input_ids_2=uncond_tokens["input_ids_2"],
        input_ids_3=uncond_tokens["input_ids_3"],
    )
    return uncond_prompt_embeds, uncond_pooled_prompt_embeds
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_optimizer(args, transformer):
    ## Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    return optimizer, lr_scheduler, params_to_optimize
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def load_if_exist(args, accelerator, dataloader_size):
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            num_update_steps_per_epoch = math.ceil(dataloader_size / args.gradient_accumulation_steps)
    else:
        initial_global_step = 0

    return initial_global_step
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_3rd_party(args, accelerator, dataloader_size):
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(dataloader_size / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(dataloader_size / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(deepcopy(args))
        # Tracker config doesn't support lists
        accelerator.init_trackers("scalewise", config=tracker_config)
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def saving(transformer, args, accelerator, global_step, logs):
    # Save by ps
    condition = logs['pick_score_coco'] > args.previous_best_ps
    if condition:
        save_path = os.path.join(args.output_dir, f"checkpoint-ps_{global_step}")
        print(f'Saving to {save_path}')
        transformer = accelerator.unwrap_model(transformer).transformer
        transformer.save_pretrained(save_path)

        # new_best = logs['pick_score']
        # args.previous_best_ps = new_best

    # save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    # accelerator.save_state(save_path)
    # logger.info(f"Saved state to {save_path}")
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
class Pipeline:
    def __init__(self, vae, transformer,
                 text_encoder, text_encoder_2, text_encoder_3,
                 tokenizer, tokenizer_2, tokenizer_3,
                 revision, variant, torch_dtype, image_processor):
        self.vae = vae
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.revision = revision
        self.variant = variant
        self.torch_dtype = torch_dtype
        self.image_processor = image_processor
# ----------------------------------------------------------------------------------------------------------------------
