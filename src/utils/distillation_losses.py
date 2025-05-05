import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from diffusers.image_processor import VaeImageProcessor

########################################################################################################################
#                            THE LOSSES NEEDED FOR THE DISTILLATION OF LLM                                             #
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
def diffusion_loss(
        transformer_llm, transformer_dm,
        prompt_embeds_dm, pooled_prompt_embeds_dm, embeds_llm, mask_llm,
        noisy_latent_image, timesteps,
        optimizer, lr_scheduler, params_to_optimize,
        accelerator, args,
        vae, global_step, do_eval=False
):
    optimizer.zero_grad(set_to_none=True)
    transformer_llm.train()
    transformer_dm.eval()

    ## STEP 1. Make a prediction with the teacher to create a target
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        model_pred_target = transformer_dm(
            noisy_latent_image,
            prompt_embeds_dm.to(torch.float16),
            pooled_prompt_embeds_dm.to(torch.float16),
            timesteps,
            return_dict=False,
        )[0]

    ## STEP 2. Make a prediction
    model_pred = transformer_llm(
            embeds_llm, mask_llm,
            noisy_latent_image
        )

    ## STEP 3. Calculate DM loss and update the generator
    loss = F.huber_loss(model_pred.float(), model_pred_target.float(), delta=1.0)
    
    ## Backpropagate
    accelerator.backward(loss)

#    check = False
#    for name, param in transformer_llm.named_parameters():
#        if 'lora' in name and param.requires_grad and not check:
#            print(param.grad.norm().item(), param.norm().item(), name)
#            check = True

    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
    avg_loss += avg_loss.item() / args.gradient_accumulation_steps

    if do_eval:
        with torch.no_grad():
            denoised_pred = noisy_latent_image - timesteps[0].item() / 1000 * model_pred
            target_pred = noisy_latent_image - timesteps[0].item() / 1000 * model_pred_target
            image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)

            latent = (denoised_pred / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(latent.half(), return_dict=False)[0]
            image = image_processor.postprocess(image, output_type='pil')
            for j, image in enumerate(image):
                image.save(f'{args.output_dir}/{global_step}_{j}_pred.jpg')

            latent = (target_pred / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(latent.half(), return_dict=False)[0]
            image = image_processor.postprocess(image, output_type='pil')
            for j, image in enumerate(image):
                image.save(f'{args.output_dir}/{global_step}_{j}_target.jpg')

    return avg_loss
# ----------------------------------------------------------------------------------------------------------------------
