import torch
from torch import nn
from transformers import (
    AutoModel,
    Gemma3nConfig,
    Gemma3nPreTrainedModel,
    GenerationMixin,
)
from transformers.models.gemma3n.modeling_gemma3n import (
    Gemma3nMultimodalEmbedder,
    Gemma3nRMSNorm,
    Gemma3nTextConfig,
    Gemma3nTextDecoderLayer,
    Gemma3nTextScaledWordEmbedding,
)


class Gemma3nTextModel(Gemma3nPreTrainedModel):
    config_class = Gemma3nTextConfig

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        # self.vocab_size = config.vocab_size

        # Gemma3n downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        if config.vocab_size:
            self.embed_tokens = Gemma3nTextScaledWordEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                embed_scale=self.config.hidden_size**0.5,
            )
        self.layers = nn.ModuleList(
            [Gemma3nTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.rotary_emb = Gemma3nTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # TODO (raushan): Fix this after RoPE refactor. For now we hack it by
        # reassigning thetas when we want to create a local RoPE layer. Config
        # defaults should hold values for global RoPE.
        # config = copy.deepcopy(config)
        # config.rope_theta = config.rope_local_base_freq
        # config.rope_scaling = {"rope_type": "default"}
        # self.rotary_emb_local = Gemma3nTextRotaryEmbedding(config=config)

        self.hidden_size = config.hidden_size
        # self.hidden_size_per_layer_input = config.hidden_size_per_layer_input

        # self.embed_tokens_per_layer = Gemma3nTextScaledWordEmbedding(
        #     config.vocab_size_per_layer_input,
        #     config.num_hidden_layers * config.hidden_size_per_layer_input,
        #     self.padding_idx,
        #     embed_scale=config.hidden_size_per_layer_input**0.5,
        # )

        # self.per_layer_model_projection = nn.Linear(
        #     self.hidden_size,
        #     config.num_hidden_layers * config.hidden_size_per_layer_input,
        #     bias=False,
        # )

        # self.per_layer_projection_norm = Gemma3nRMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)

        # self.altup_projections = nn.ModuleList(
        #     [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]
        # )

        # self.altup_unembed_projections = nn.ModuleList(
        #     [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]
        # )

        # self.register_buffer("per_layer_projection_scale", torch.tensor(self.hidden_size**-0.5), persistent=False)
        # self.register_buffer("per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


class Gemma3nModel(Gemma3nPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    # we are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
    accepts_loss_kwargs = False

    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config=config.vision_config)
        self.vocab_size = config.text_config.vocab_size

        language_model = Gemma3nTextModel(config=config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.vocab_size_per_layer_input = config.text_config.vocab_size_per_layer_input
        # self.audio_tower = AutoModel.from_config(config.audio_config)
        self.embed_vision = Gemma3nMultimodalEmbedder(config.vision_config, config.text_config)
        # self.embed_audio = Gemma3nMultimodalEmbedder(config.audio_config, config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values, do_pooling=False, return_dict=True
        ).last_hidden_state
        # Convert from (batch, channels, height, width) to (batch, height * width, channels) where:
        # height == width and height * width == Gemma3nConfig.vision_soft_tokens_per_image.
        vision_outputs = vision_outputs.reshape(
            vision_outputs.shape[0],
            self.config.vision_config.hidden_size,
            self.config.vision_soft_tokens_per_image,
        ).permute(0, 2, 1)
        # Normalize and embed the soft tokens into language model space.
        vision_outputs *= self.config.vision_config.hidden_size**0.5
        return self.embed_vision(inputs_embeds=vision_outputs)


class Gemma3nForConditionalGeneration(Gemma3nPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    base_model_prefix = "model"

    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        self.model = Gemma3nModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(self, pixel_values):
        return self.model.get_image_features(pixel_values)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        raise AttributeError("Use embed_vision instead of multi_modal_projector.")
