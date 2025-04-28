import torch
from tqdm import tqdm

########################################################################################################################
#                                            SAMPLER UTILS                                                             #
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
class FlowMatchingSolver:

    ## ---------------------------------------------------------------------------
    def __init__(
        self,
        noise_scheduler,
    ):
        self.noise_scheduler = noise_scheduler
    ## ---------------------------------------------------------------------------


    ## ---------------------------------------------------------------------------
    def flow_matching_single_step(self, sample, model_output, sigma, sigma_next):
        prev_sample = sample + (sigma_next - sigma) * model_output
        return prev_sample
    ## ---------------------------------------------------------------------------


    ## ---------------------------------------------------------------------------
    @torch.no_grad()
    def flow_matching_sampling(
        self,
        model,
        latent,
        embeds_llm, mask_llm,
        idx_start, idx_end,
        sigmas=None,
    ):
        sigmas = self.noise_scheduler.sigmas if sigmas is None else sigmas
        while True:
            sigma = sigmas[idx_start].to(device=latent.device)
            sigma_next = sigmas[idx_start + 1].to(device=latent.device)

            with torch.autocast("cuda", dtype=torch.float16):
                noise_pred = model(
                    embeds_llm,
                    mask_llm,
                    latent
                )

            latent = self.flow_matching_single_step(latent, noise_pred,
                                                    sigma[:, None, None, None],
                                                    sigma_next[:, None, None, None])

            if (idx_start + 1)[0].item() == idx_end[0].item():
                break
            idx_start = idx_start + 1

        return latent
    ## ---------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
