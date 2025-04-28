import copy
import torch
import torch.nn as nn

from diffusers.utils import USE_PEFT_BACKEND, unscale_lora_layers, is_torch_version, logging, scale_lora_layers

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


########################################################################################################################
#                                           LLM+DM TRANSFORMER                                                         #
########################################################################################################################


# Transformer with an additional layers for image tokens (adapters)
# ----------------------------------------------------------------------------------------------------------------------
class TransformerLLM(nn.Module):
    def __init__(self, transformer_llm, pos_embed_layer_dm, inner_dim_dm):
        super().__init__()
        self.model = transformer_llm.model
        self.model.lm_head = torch.nn.Identity()  # Drop an unnecessary layer
        self.model.embed_tokens = torch.nn.Identity() # Drop an unnecessary layer
        self.patch_size = 2
        self.out_channels = 16
        self.size = 128

        self.adapter_layer = nn.Sequential(
            nn.Linear(inner_dim_dm, transformer_llm.config.hidden_size),
            copy.deepcopy(transformer_llm.model.model.layers[0].input_layernorm),
            nn.GELU(),
            nn.Linear(transformer_llm.config.hidden_size, transformer_llm.config.hidden_size),
            nn.LayerNorm(transformer_llm.config.hidden_size),
            nn.GELU(),
            nn.Linear(transformer_llm.config.hidden_size, transformer_llm.config.hidden_size),
            copy.deepcopy(transformer_llm.model.model.layers[0].post_attention_layernorm),
        )
        self.pos_embed_layer_dm = copy.deepcopy(pos_embed_layer_dm)
        self.proj_out = nn.Linear(transformer_llm.config.hidden_size,
                                  self.patch_size * self.patch_size * 16, bias=True)

        self.pos_embed_layer_dm.requires_grad_(False)
        self.adapter_layer.requires_grad_(True)
        self.proj_out.requires_grad_(True)

    def forward(self, embeddings_text, mask_llm, latent_image):
        # Create unified embeddings
        embeddings_image = self.pos_embed_layer_dm(latent_image) # [b x 16 x 128 x 128] -> [b x 4096 x 2432]
        embeddings_image = self.adapter_layer(embeddings_image) # [b x 4096 x 2432] -> [b x 4096 x 4096]
        embeddings = torch.cat([embeddings_text, embeddings_image], dim=1) #  [b x (text + 4096) x 4096]

        # Create a unified attention mask. The text does not participate in a calculation.
        b_size, seq_len_text,  seq_len_img = embeddings.shape[0], embeddings_text.shape[1], embeddings_image.shape[1]
        attention_mask = torch.ones(b_size, seq_len_img).cuda()
        attention_mask = torch.cat((mask_llm, attention_mask), dim=1)

        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        last_hidden_states = outputs.logits
        seq_idx = last_hidden_states.shape[1] - 4096
        hidden_states = last_hidden_states[:, seq_idx:, :]
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.patch_size
        height = self.size // patch_size
        width = self.size // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        return output
# ----------------------------------------------------------------------------------------------------------------------
