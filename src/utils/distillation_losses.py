import torch
import torch.nn.functional as F


########################################################################################################################
#                            THE LOSSES NEEDED FOR THE DISTILLATION OF LLM                                             #
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
def diffusion_loss(
        transformer_llm, transformer_dm,
        prompt_embeds_dm, pooled_prompt_embeds_dm, embeds_llm,
        noisy_latent_image, timesteps,
        optimizer, lr_scheduler, params_to_optimize,
        accelerator, args,
):
    optimizer.zero_grad(set_to_none=True)
    transformer_llm.train()
    transformer_dm.eval()

    ## STEP 1. Make a prediction with the teacher to create a target
    with torch.no_grad():
        model_pred_target = transformer_dm(
            noisy_latent_image,
            prompt_embeds_dm,
            pooled_prompt_embeds_dm,
            timesteps,
            return_dict=False,
        )[0]

    ## STEP 2. Make a prediction with the student to create a prediction
    model_pred = transformer_llm(
        embeds_llm,
        noisy_latent_image,
    )[0]

    ## STEP 3. Calculate DM loss and update the generator
    c = 0.0
    loss = torch.sqrt((model_pred.float() - model_pred_target.float()) ** 2 + c ** 2) - c
    loss = torch.mean(loss)
        
    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
    avg_loss += avg_loss.item() / args.gradient_accumulation_steps

    ## Backpropagate
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    return avg_loss
# ----------------------------------------------------------------------------------------------------------------------
