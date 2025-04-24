import torch
from accelerate.logging import get_logger
from pathlib import Path

from src.utils.train_utils import distributed_sampling
from src.utils.flow_matching_sampler import FlowMatchingSolver
from src.pipelines.stable_diffusion_3 import ScaleWiseStableDiffusion3Pipeline
from src.utils.train_utils import unwrap_model
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from src.utils.metrics import calculate_scores

logger = get_logger(__name__)


@torch.no_grad()
def validate_teacher(args):

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = prepare_accelertor(args, logging_dir)
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    ).to(torch.float16)

    _, _, _, vae, \
    text_encoder, text_encoder_2, text_encoder_3, \
    tokenizer, tokenizer_2, tokenizer_3, \
    noise_scheduler, weight_dtype = prepare_models(args, accelerator)
    transformer = accelerator.prepare(transformer)

    noise_scheduler.set_timesteps(args.num_boundaries)
    fm_solver = FlowMatchingSolver(noise_scheduler, args.num_boundaries, args.scales, args.boundaries)

    pipeline = ScaleWiseStableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        transformer=unwrap_model(transformer, accelerator),
        text_encoder=unwrap_model(text_encoder, accelerator),
        text_encoder_2=unwrap_model(text_encoder_2, accelerator),
        text_encoder_3=unwrap_model(text_encoder_3, accelerator),
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    for eval_set_name in ['mjhq', 'coco']:
        eval_prompts_path = f'prompts/{eval_set_name}.csv'
        if eval_set_name == "coco":
            fid_stats_path = args.coco_ref_stats_path
        else:
            fid_stats_path = args.mjhq_ref_stats_path
        images, prompts = distributed_sampling(pipeline, args, eval_prompts_path,
                                               prepare_prompt_embed_from_caption, fm_solver, noise_scheduler,
                                               accelerator, logger, cfg_scale=args.cfg_teacher)


        if accelerator.is_main_process:
            image_reward, pick_score, clip_score, fid_score = calculate_scores(
                args,
                images,
                prompts,
                ref_stats_path=fid_stats_path,
            )
            logs = {
                f"fid_{eval_set_name}": fid_score.item(),
                f"pick_score_{eval_set_name}": pick_score.item(),
                f"clip_score_{eval_set_name}": clip_score.item(),
                f"image_reward_{eval_set_name}": image_reward.item(),
            }
            print(logs)