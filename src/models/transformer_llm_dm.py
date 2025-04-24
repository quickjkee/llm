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
        self.model.lm_head = None  # Drop an unnecessary layer
        self.model.embed_tokens = None # Drop an unnecessary layer

        self.first_linear_layer = nn.Linear(inner_dim_dm, transformer_llm.config.hidden_size)
        self.adapter_layer = nn.Sequential(
            self.first_linear_layer,
            copy.deepcopy(transformer_llm.model.model.layers[0]),
        )
        self.pos_embed_layer_dm = pos_embed_layer_dm

        self.pos_embed_layer_dm.requires_grad_(False)
        self.first_linear_layer.requires_grad_(True)

    def forward(self, embeddings_text, latent_image):
        # Create unified embeddings
        embeddings_image = self.pos_embed_layer_dm(latent_image) # [b x 16 x 128 x 128] -> [b x 4096 x 2432]
        embeddings_image = self.adapter_layer(embeddings_image) # [b x 4096 x 2432] -> [b x 4096 x 4096]
        embeddings = torch.cat([embeddings_text, embeddings_image], dim=1) #  [b x (text + 4096) x 4096]

        # Create a unified attention mask. The text does not participate in a calculation.
        b_size, seq_len_text,  seq_len_img = embeddings.shape[0], embeddings_text.shape[1], embeddings.shape[1]
        attention_mask_text = torch.zeros(b_size, seq_len_text, seq_len_text + seq_len_img)
        attention_mask_imgs = torch.ones(b_size, seq_len_img, seq_len_img + seq_len_text)
        attention_mask = torch.cat([attention_mask_text, attention_mask_imgs], dim=1)

        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state

        return last_hidden_states
# ----------------------------------------------------------------------------------------------------------------------
