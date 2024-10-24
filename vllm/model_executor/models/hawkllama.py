
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING
from typing import List, Optional, Union
import itertools
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType
from transformers import AutoImageProcessor,SiglipVisionConfig,CLIPVisionConfig
from PIL import Image

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.image_processing_utils import select_best_resolution
from transformers.modeling_outputs import ModelOutput
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.auto import AutoModel, AutoModelForCausalLM

from einops import rearrange


from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, BatchedTensors
from vllm.sequence import IntermediateTensors, SamplerOutput


from .interfaces import SupportsVision
from .llava import LlavaMultiModalProjector
from .clip import (CLIPVisionModel, dummy_image_for_clip,
                   dummy_seq_data_for_clip, get_clip_image_feature_size,
                   get_clip_patch_grid_length, input_processor_for_clip)

from .siglip import (SiglipVisionModel, dummy_image_for_siglip,
                     dummy_seq_data_for_siglip, get_siglip_image_feature_size,
                     get_siglip_patch_grid_length, input_processor_for_siglip)
from .utils import (filter_weights, init_vllm_registered_model,
                    merge_vision_embeddings)

logger = init_logger(__name__)

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.lm_head": "lm_head",
    "language_model.model": "language_model",
}

MAX_IMAGE_FEATURE_SIZE_HEIGHT = MAX_IMAGE_FEATURE_SIZE_WIDTH = 384



class LlavaNextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaNextForConditionalGeneration`]. It is used to instantiate an
    Llava-NeXT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
    model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
            If `"full"`, the full vision features are used.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        image_grid_pinpoints (`List`, *optional*, defaults to `[[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]`):
            A list of possible resolutions to use for processing high resolution images. Each item in the list should be a tuple or list
            of the form `(height, width)`.

    Example:

    ```python
    >>> from transformers import LlavaNextForConditionalGeneration, LlavaNextConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Llava-Next llava-hf/llava-v1.6-mistral-7b-hf style configuration
    >>> configuration = LlavaNextConfig(vision_config, text_config)

    >>> # Initializing a model from the llava-hf/llava-v1.6-mistral-7b-hf style configuration
    >>> model = LlavaNextForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llava_next"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=128265,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="full",
        vision_feature_layer=-2,
        image_grid_pinpoints=None,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        image_grid_pinpoints = (
            image_grid_pinpoints
            if image_grid_pinpoints is not None
            else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
        )
        self.image_grid_pinpoints = image_grid_pinpoints

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "siglip_vision_model"
            )
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["siglip_vision_model"](
                hidden_act="gelu_pytorch_tanh",
                intermediate_size=4304,
                hidden_size=1152,
                patch_size=14,
                image_size=384,
                num_hidden_layers=27,
                num_attention_heads=16,
                layer_norm_eps=1e-06
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config

        super().__init__(**kwargs)


class LlavaNextProcessor(ProcessorMixin):
    r"""
    Constructs a LLaVa-NeXT processor which wraps a LLaVa-NeXT image processor and a LLaMa tokenizer into a single processor.

    [`LlavaNextProcessor`] offers all the functionalities of [`LlavaNextImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~LlavaNextProcessor.__call__`] and [`~LlavaNextProcessor.decode`] for more information.

    Args:
        image_processor ([`LlavaNextImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["tokenizer"]
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None):            
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        image_aspect_ratio='pad',
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        LlavaNextImageProcessor's [`~LlavaNextImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if images is not None:
            image_inputs = self.process_images(images, return_tensors, image_aspect_ratio)
        else:
            image_inputs = {}
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length, add_special_tokens=True,
        )

        return BatchFeature(data={**text_inputs, **image_inputs})

    def process_images(self, image_path, return_tensors, image_aspect_ratio='pad'):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        if image_aspect_ratio == 'keep':
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            max_len, min_len = 448, 224
            shortest_edge = int(min(max_len / aspect_ratio, min_len))
            image = self.image_processor.preprocess(
                image, return_tensors=return_tensors, size={"shortest_edge": shortest_edge}
            )['pixel_values'][0]
        elif image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(0 for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        output = {
            'pixel_values' : image.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
        }
        return output
        
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
    
    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        args = [AutoImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)]
        args.extend(super()._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs))
        return args


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).

    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor




# Copied from transformers.models.llava.modeling_llava.LlavaMultiModalProjector with Llava->LlavaNext
class LlavaNextMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaNextConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.downsample = nn.Conv2d(
            in_channels=config.vision_config.hidden_size,
            out_channels=config.vision_config.hidden_size,
            kernel_size=3, 
            stride=2,
            padding=1
        )
        self.alpha = nn.Parameter(torch.tensor(0.))
        

    def forward(self, image_features):
        image_features = rearrange(image_features, "(b n_tile) n d -> b n_tile n d ", n_tile=5)
        sml_images = image_features[:, 0]
        sub_images = image_features[:, 1:]
        h = int(image_features.shape[-2] ** 0.5)
        sub_images = rearrange(sub_images, "b (hs ws) (h w) d -> b d (hs h) (ws w)", hs=2, h=h)
        sub_images = self.downsample(sub_images)
        sub_images = rearrange(sub_images, "b d h w -> b (h w) d")
        x = sml_images + self.alpha * sub_images
        hidden_states = self.linear_1(x)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


LLAVA_NEXT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlavaNextConfig`] or [`LlavaNextVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAVA_NEXT_START_DOCSTRING,
)
# Copied from transformers.models.llava.modeling_llava.LlavaPreTrainedModel with Llava->LlavaNext,llava->llava_next
class LlavaNextPreTrainedModel(PreTrainedModel):
    config_class = LlavaNextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaNextVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        # important: this ported version of LlavaNext isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava_next should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa


LLAVA_NEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`LlavaNextImageProcessor.__call__`] for details. [`LlavaProcessor`] uses
            [`LlavaNextImageProcessor`] for processing images.
        image_sizes (`torch.LongTensor` of shape `(batch_size, 2)`, *optional*):
            The sizes of the images in the batch, being (height, width) for each image.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
            If `"full"`, the full vision features are used.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""




def _get_llava_next_num_unpadded_features(
    original_height: int,
    original_width: int,
    npatches: int,
    num_patch_height: int,
    num_patch_width: int,
) -> Tuple[int, int]:
    current_height = npatches * num_patch_height
    current_width = npatches * num_patch_width

    aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if aspect_ratio > current_aspect_ratio:
        new_height = (original_height * current_width) // original_width
        padding = (current_height - new_height) // 2
        current_height -= padding * 2
    else:
        new_width = (original_width * current_height) // original_height
        padding = (current_width - new_width) // 2
        current_width -= padding * 2

    unpadded_features = current_height * current_width
    newline_features = current_height
    return (unpadded_features, newline_features)

def get_llava_next_image_feature_size(
    hf_config: LlavaNextConfig,
    *,
    input_height: int,
    input_width: int,
) -> int:
    config_dict = vars(hf_config)  # or hf_config.__dict__
    vision_config = hf_config.vision_config

    if isinstance(vision_config, CLIPVisionConfig):
        num_patches = get_clip_patch_grid_length(
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
        )
        base_feature_size = get_clip_image_feature_size(vision_config)
    elif isinstance(vision_config, SiglipVisionConfig):
        num_patches = get_siglip_patch_grid_length(
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
        )
        base_feature_size = get_siglip_image_feature_size(vision_config)
    else:
        msg = f"Unsupported vision config: {type(vision_config)}"
        raise NotImplementedError(msg)

    strategy = hf_config.vision_feature_select_strategy
    if strategy == "default":
        base_feature_size -= 1
    elif strategy == "full":
        pass
    else:
        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    num_patch_height, num_patch_width = get_anyres_image_grid_shape(
        image_size=(input_height, input_width),
        grid_pinpoints=hf_config.image_grid_pinpoints,
        patch_size=vision_config.image_size,
    )

    (
        unpadded_feature_size,
        newline_feature_size,
    ) = _get_llava_next_num_unpadded_features(input_height, input_width,
                                              num_patches, num_patch_height,
                                              num_patch_width)

    # return unpadded_feature_size + newline_feature_size + base_feature_size
    return 729


def get_max_llava_next_image_tokens(ctx: InputContext):

    return get_llava_next_image_feature_size(
        LlavaNextConfig(),
        input_height=MAX_IMAGE_FEATURE_SIZE_HEIGHT,
        input_width=MAX_IMAGE_FEATURE_SIZE_WIDTH,
    )


#需修改
def dummy_data_for_llava_next(ctx: InputContext, seq_len: int):
    hf_config = LlavaNextConfig()
    vision_config = hf_config.vision_config
    image_feature_size=get_max_llava_next_image_tokens(ctx)
    if isinstance(vision_config, SiglipVisionConfig):
        seq_data = dummy_seq_data_for_siglip(
            vision_config,
            seq_len,
            image_token_id=hf_config.image_token_index,
            image_feature_size_override=image_feature_size,
        )

        mm_data = dummy_image_for_siglip(
            vision_config,
            image_width_override=MAX_IMAGE_FEATURE_SIZE_WIDTH,
            image_height_override=MAX_IMAGE_FEATURE_SIZE_HEIGHT,
        )

        return seq_data, mm_data

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


def input_processor_for_llava_next(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config
    hf_config = model_config.hf_config
    vision_config = hf_config.vision_config

    image_data = multi_modal_data["image"]
    if isinstance(image_data, Image.Image):
        width, height = image_data.size

        image_feature_size = get_llava_next_image_feature_size(
            hf_config,
            input_height=height,
            input_width=width,
        )
    elif isinstance(image_data, torch.Tensor):
        raise NotImplementedError("Embeddings input is not supported yet")
    else:
        raise TypeError(f"Invalid image type: {type(image_data)}")

    vision_config = hf_config.vision_config

    if isinstance(vision_config, SiglipVisionConfig):
        return input_processor_for_siglip(
            model_config,
            vision_config,
            llm_inputs,
            image_token_id=hf_config.image_token_index,
            image_feature_size_override=image_feature_size,
        )

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)

@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_llava_next_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_llava_next)
@INPUT_REGISTRY.register_input_processor(input_processor_for_llava_next)
class HawkLlavaNextForConditionalGeneration(nn.Module, SupportsVision):
    def __init__(self,
                 config: LlavaNextConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
      
  

        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config

        # Initialize the vision tower only up to the required feature layer
        # vision_feature_layer = config.vision_feature_layer
        # if vision_feature_layer < 0:
        #     num_hidden_layers = config.vision_config.num_hidden_layers \
        #         + vision_feature_layer + 1
        # else:
        #     num_hidden_layers = vision_feature_layer + 1

        # TODO: 改成siglip
        self.vision_tower = SiglipVisionModel(
            config.vision_config, num_hidden_layers_override=config.vision_config.num_hidden_layers)
        self.multi_modal_projector = LlavaNextMultiModalProjector(config)
        
        self.language_model = init_vllm_registered_model(config.text_config, cache_config, quant_config)
        self.unpadded_vocab_size = config.text_config.vocab_size
        # self.lm_head = ParallelLMHead(
        #     self.unpadded_vocab_size,
        #     config.text_config.hidden_size,
        #     org_num_embeddings=self.language_model.org_vocab_size,
        #     quant_config=quant_config)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.text_config.vocab_size,
                                                logit_scale)
        self.sampler = Sampler()


    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> SamplerOutput:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
        ```"""

        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        vision_feature_layer = -2
        vision_feature_select_strategy = "full"

        if pixel_values is not None:
          
            inputs_embeds = self.language_model.model.get_input_embeddings(input_ids)
           
            # assert pixel_values.ndim == 6, "pixel_values should be of shape (b, T_img, F, C, H, W)"
                
            # batch_size, T, F, num_channels, height, width = pixel_values.shape
            _,num_channels,height,width=pixel_values.shape
            # assert F == 1, "Only single frame supported"

            # pixel_values = rearrange(pixel_values, "b T F c h w -> (b T F) c h w")
            h = w = height // 2
            sml_images = torch.nn.functional.interpolate(pixel_values, size=(h, w), mode='bicubic')
            sub_images = rearrange(pixel_values, "b c (hs h) (ws w) -> (b hs ws) c h w", hs=2, ws=2)
            reshaped_pixel_values = torch.cat([sml_images,sub_images])
            image_features = self.vision_tower(reshaped_pixel_values)

            selected_image_feature = image_features

            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature

            image_features = self.multi_modal_projector(selected_image_feature)
          

            # split up image_features for each of the individual images
            # hence we get a list of image_features, each of shape (5, num_patches, hidden_size)
            # if we assume each image has 5 image features (base image + 4 patches)
            # split_sizes = [image.shape[0] for image in pixel_values]
            split_sizes = [1 for image in pixel_values]
            # image_features = torch.load('media_features.pt').to(image_features.device)[0,0]
            image_features = torch.split(image_features, split_sizes, dim=0)

            # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
            height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]

                    if height * width != base_image_feature.shape[0]:
                        raise ValueError("The number of patches is not consistent with the image size.")
                    num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                        image_sizes[image_idx],
                        self.config.image_grid_pinpoints,
                        self.config.vision_config.image_size,
                    )
                    image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                    image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                    image_feature = unpad_image(image_feature, image_sizes[image_idx])
                    image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature[0]
                new_image_features.append(image_feature)
            image_features = torch.stack(new_image_features, dim=0)

            # inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
            #     image_features, inputs_embeds, input_ids, attention_mask, labels
            # )

            
            inputs_embeds = merge_vision_embeddings(
                input_ids, inputs_embeds, image_features,
                self.config.image_token_index)

            input_ids = None
        else:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            None,
            inputs_embeds=inputs_embeds
        )
        return hidden_states


    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.language_model.compute_logits(hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights):
        # only doing this for language model part for now.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # if "lm" in name:
            #     self.lm_head.weight_loader(self.lm_head.weight, loaded_weight)
            #     continue
            # if name.startswith("language_model"):
            #     name = name.replace("language_model.model.", "language_model.",
            #                         1)
            # 没有用到siglip的 最后一层
            # if "vision_tower.vision_model.encoder.layers.26" in name:
            #     continue
            if "language_model" not in name:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # if name not in params_dict:
                name = name.replace(weight_name, param_name)
                
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

           


