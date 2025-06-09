
import json
import logging
import math
import os
import types
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from threading import Thread
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Type
from typing import Any 
from typing import Dict 
from typing import Callable

import numpy as np
import soundfile as sf
import torch
from torch import Tensor 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm
from transformers import AutoProcessor
from transformers import BertTokenizerFast
from transformers import LlamaConfig
from transformers import LlamaModel
from transformers import LogitsProcessor
from transformers import PreTrainedModel
from transformers import Qwen2ForCausalLM
from transformers import Qwen2PreTrainedModel
from transformers import TextIteratorStreamer
from transformers import TopKLogitsWarper
from transformers import TopPLogitsWarper
from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache
from transformers.cache_utils import EncoderDecoderCache
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import ModelOutput
from transformers.models.whisper.modeling_whisper import ACT2FN
from transformers.models.whisper.modeling_whisper import WHISPER_ATTENTION_CLASSES
from transformers.models.whisper.modeling_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

try:
    from vector_quantize_pytorch import GroupedResidualFSQ
    from vocos import Vocos
    from vocos.pretrained import instantiate_class

    _tts_deps = True
except:
    _tts_deps = False

from .configuration_khaanh import ConditionalChatTTSConfig
from .configuration_khaanh import KhaanhConfig
from .modeling_navit_siglip import SiglipVisionTransformer
from .resampler import Resampler 
from .utils import NumberToTextConverter 
from .utils import sentence_end 
from .utils import VoiceChecker 

logger = logging.getLogger(__name__)


@dataclass
class OmniOutput(ModelOutput):
    """
    Cấu trúc dữ liệu đầu ra cho mô hình AI đa phương thức, chứa cả 
    kết quả văn bản và âm thanh.

    Attributes:
        text (Optional[Union[str, List[str], Iterator]]): 
            Văn bản được mô hình sinh ra hoặc xử ký. Có thể ở dạng:
                - Một chuỗi đơn (str), 
                - Danh sách các chuỗi (List[str]) nếu xử lý theo batch, 
                - Hoặc iterator dùng trong trường hợp sinh văn bản dạng dòng (streaming). 
        
        spk_embeds (Optional[torch.FloatTensor]):
            Speaker embedding (biểu diễn đặc trung của người nói) dưới dạng tensor 
                Pytorch. Thường dùng trong các mô hình tổng hợp giọng nói hoặc clone giọng.
        
        audio_wav (Optional[np.ndarray]): 
            Dữ liệu sóng âm thanh (dạng waveform) đầu ra, được biểu diễn bằng mảng NumPy.
            Thường là mảng 1 chiều (mono) hoặc 2 chiều (nếu multi-channel).

        sampling_rate (Optional[int]): 
            Tần số lấy mẫu của tín hiệu âm thanh đầu ra (đơn vị: Hz).
            Ví dụ phổ biến: 16000Hz, 22050Hz, 44100Hz,...
    """
    text: Optional[Union[str, List[str], Iterator]] = None # Văn bản đầu ra từ mô hình
    spk_embeds: Optional[torch.FloatTensor] = None # Tensor embedding người nói
    audio_wav: Optional[np.ndarray] = None # Sóng âm thanh đầu ra dạng numpy
    sampling_rate: Optional[int] = None # Tần số lấy mẫu của audio



class KhaanhPreTrainedModel(Qwen2PreTrainedModel):
    """
    Lớp nền (base class) cho tất cả các mô hình được huấn luyện sẵn. 

    Kế thừa từ:
        Qwen2PreTrainedModel: lớp mô hình huấn luyện sẵn từ kiến trúc. 
    
    Attributes:
        config_class (Type[KhaanhConfig]): Kiểu cấu hình được sử dụng 
            cho mô hình này. Giúp HuggingFace tự động ánh xạ khi load 
            mô hình từ checkpoint.
    """
    config_class: Type[KhaanhConfig] = KhaanhConfig



# Copied from transformers.models.whisper.modeling_whisper.WhisperEncoderLayer and add use_cache for streaming inference
class KhaanhWhisperEncoderLayer(nn.Module):
    """
    Một lớp encoder trong mô hình mã hóa âm thanh Whisper. Bao gồm 
        (1) Attention chuẩn với mask;
        (2) LayerNorm trước Attention và FFN (pre-norm);
        (3) Feed-Forward Network (2 linear + activation);
        (4) Dropout & residual connection;
    """
    def __init__(self, config: WhisperConfig, layer_idx: Optional[int] = None) -> None:
        """
        Khởi tạo một lớp encoder layer.

        Args:
            config (WhisperConfig): Cấu hình mô hình Whisper.
            layer_idx (int, optional): Chỉ số của lớp (dùng cho cache hoặc debug).
        """
        super().__init__()
        self.embed_dim: int = config.d_model

        # Self-attention (hỗ trợ FlashAttention hoặc kiểu khác tuỳ config)
        self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            layer_idx=layer_idx,
        )

        # Chuẩn hoá trước attention
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # Cấu hình dropout
        self.dropout = config.dropout
        self.activation_dropout = config.activation_dropout

        # Hàm kích hoạt (ReLU, GELU,...)
        self.activation_fn = ACT2FN[config.activation_function]

        # Feedforward (2 tầng)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)

        # LayerNorm cuối 
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
            self,
            hidden_states: torch.Tensor, # (batch_size, seq_len, embed_dim)
            attention_mask: torch.Tensor, # (batch_size, 1, tgt_len, src_len)
            layer_head_mask: torch.Tensor, # (encoder_attention_heads, )
            output_attentions: bool = False,
            past_key_values: Optional[EncoderDecoderCache] = None,
            use_cache: Optional[bool] = False,
        ) -> torch.Tensor:
        r"""
        Execute one step through encoder layer. 

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, embed_dim)`):
                Hidden states to be fed into the encoder layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, 1, tgt_len, src_len)`):
                Attention mask where padding elements are indicated by large negative values.
            layer_head_mask (`torch.FloatTensor` of shape `(encoder_attention_heads,)`):
                Mask to nullify selected heads of the attention modules.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attention weights.
            past_key_values (`EncoderDecoderCache`, *optional*):
                Past key-value pairs used for incremental decoding.
            use_cache (`bool`, *optional*):
                Whether or not to return updated `past_key_values` for caching.
        Returns:
            A tuple of shape `(hidden_states, optional(attn_weights), optional(past_key_values))`.
        """
        # Attention block
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, attn_weights, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_values,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states  # Residual connection

        # Feed-forward block
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states  # Residual connection

        # Xử lý giá trị vô hạn hoặc NaN (tránh lỗi FP16)
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        # Trả về kết quả 
        outputs: Tuple[torch.Tensor, ...] = (hidden_states, )

        if output_attentions: outputs += (attn_weights, )

        if use_cache: outputs += (past_key_values, )

        return outputs 
        


# Copied from from transformers.models.whisper.modeling_whisper.WhisperEncoder and add use_cache for streaming inference
class KhaanhWhisperEncoder(WhisperEncoder):
    """
    Audio encoder model kế thừa từ WhisperEncoder, với các lớp 
    encoder được thay thế bằng KhaanhWhisperEncoderLayer tùy biến. 
    """
    def __init__(self, config: WhisperConfig) -> None:
        """
        Khởi tạo encoder gồm nhiều lớp encoder layer tùy biến. 

        Args:
            config (WhisperConfig): Cấu hình mô hình Whisper, chứa thông tin về số lớp, dropout, attention, ... 
        """
        super().__init__(config) 
        # Khởi tạo danh sách các lớp encoder (dạng ModuleList) 
        self.layers: nn.ModuleList = nn.ModuleList([
            KhaanhWhisperEncoderLayer(config, layer_idx=i) for i in range(config.encoder_layers) 
        ])
    
    def forward(
            self,
            input_features: Tensor, # (B, feature_dim, seq_len)
            attention_mask: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            past_key_values: Optional[EncoderDecoderCache] = None,
            use_cache: Optional[bool] = None,
        ) -> Union[BaseModelOutputWithPast, Tuple]:
        r"""
        Forward pass of the Whisper encoder.

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of log-mel features extracted from the raw audio waveform. Typically generated
                by a feature extractor (e.g., `WhisperFeatureExtractor`) that processes `.flac` or `.wav`
                files into padded 2D mel spectrogram frames. These features are projected via convolution layers
                (`conv1` and `conv2`) and then transformed into embeddings for the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                Not used by Whisper for masking `input_features`, but included for API compatibility with
                other models. If provided, it is simply ignored within the model. By default, Whisper
                effectively ignores silence in the input log-mel spectrogram.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected attention heads. The elements should be either 1 or 0, where:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked** (i.e., the attention head is dropped).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attention tensors of all encoder layers. If set to `True`, the
                returned tuple (or `BaseModelOutputWithPast`) will contain an additional element with
                attention weights for each encoder layer.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. If set to `True`, the returned
                tuple (or `BaseModelOutputWithPast`) will contain a tuple of hidden states, including the
                initial embedding output as well as the outputs of each layer.
            return_dict (`bool`, *optional*):
                Whether or not to return a `BaseModelOutputWithPast` (a subclass of `ModelOutput`) instead
                of a plain tuple. If set to `True`, the output will be a `BaseModelOutputWithPast` object,
                otherwise it will be a tuple.
            past_key_values (`EncoderDecoderCache`, *optional*):
                When using caching for faster inference, this is an object that stores the key-value pairs
                for attention states. If provided, the model will append new states to the existing cache
                and return the updated cache. This speeds up sequential decoding or chunked inference.
                - If `past_key_values` is `None`, no past states are used or returned.
                - If `past_key_values` is not `None` and `use_cache=True`, the model will use the provided
                cache and return the updated cache (as `next_encoder_cache`).
            use_cache (`bool`, *optional*):
                Whether or not the model should use caching (`past_key_values`) to speed up processing
                during inference. When set to `True`, the model will:
                - Inspect and use `past_key_values` if provided.
                - Return updated `past_key_values` (under the name `next_encoder_cache` in
                    `BaseModelOutputWithPast`).
        Returns:
            `BaseModelOutputWithPast` or `tuple` (depending on `return_dict`):
                If `return_dict=True`, a `BaseModelOutputWithPast` is returned, which contains:
                - **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The output of the final encoder layer.
                - **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned if `output_hidden_states=True`):
                Hidden states of the model at each layer (including the initial projection).
                - **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned if `output_attentions=True`):
                Attention weights from each encoder layer.
                - **past_key_values** (an object of type `EncoderDecoderCache` or `None`, *optional*):
                Updated cache of key-value pairs if `use_cache=True`.
                If `return_dict=False`, a tuple is returned, where the format is:
                `(last_hidden_state, hidden_states, attentions)`, with `hidden_states` and `attentions`
                only present if their respective `output_*` arguments are set to `True`.
        Example:
            >>> from transformers import AutoFeatureExtractor, WhisperConfig, WhisperForConditionalGeneration
            >>> import torch
            >>> # Load a feature extractor and a Whisper model
            >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
            >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
            >>> # Assume you have audio (list of floats or numpy array) loaded from a file
            >>> # Then extract the mel features:
            >>> input_features = feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
            >>> # Forward pass
            >>> outputs = model.encoder(
            ...     input_features=input_features,
            ...     output_hidden_states=True,
            ...     output_attentions=True,
            ...     use_cache=True
            ... )
            >>> # Retrieve the last hidden state
            >>> last_hidden_state = outputs.last_hidden_state
            >>> print(last_hidden_state.shape)
            torch.Size([batch_size, seq_length, hidden_size])
            >>> # Retrieve the intermediate hidden states if output_hidden_states=True
            >>> all_encoder_hidden_states = outputs.hidden_states
            >>> # Retrieve attention weights if output_attentions=True
            >>> all_encoder_attentions = outputs.attentions
            >>> # Retrieve updated past key values if use_cache=True
            >>> encoder_cache = outputs.past_key_values
        """
        # Đảm bảo các flag được lấy từ config nếu không được truyền vào
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ép kiểu và thiết bị của input về giống với layer conv1
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        # Trích xuất đặc trưng qua 2 tầng conv + GELU
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))  # (B, D1, T1)
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))   # (B, D2, T2)

        # Đưa về định dạng [B, T, D] cho transformer encoder
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # (B, T, D)

        # Lấy embedding vị trí (positional embeddings)
        embed_pos = self.embed_positions.weight  # (max_pos, D)

        past_key_values_length = 0

        # Xử lý positional embedding nếu có cache
        if use_cache:
            if past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif isinstance(past_key_values, list):
                # Trường hợp legacy format: chuyển về DynamicCache
                past_key_values = EncoderDecoderCache(DynamicCache.from_legacy_cache(past_key_values), DynamicCache())
            elif isinstance(past_key_values, DynamicCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())

            # Tính độ dài cache hiện tại để dịch positional embedding
            past_key_values_length = past_key_values.self_attention_cache.get_usable_length(inputs_embeds.shape[1])

            # Nếu audio quá dài, lặp lại phần positional cuối cùng
            if inputs_embeds.shape[1] + past_key_values_length > embed_pos.shape[0]:
                logger.warning("seems the audio is longer than 30s. repeating the last part of the audio")

                embed_pos_front = embed_pos[past_key_values_length:, :]
                embed_pos = torch.cat((
                    embed_pos_front,
                    torch.repeat_interleave(
                        embed_pos[-1, :].unsqueeze(0),
                        inputs_embeds.shape[1] - embed_pos.shape[0] + past_key_values_length,
                        dim=0,
                    ),
                ))
            else:
                # Cắt đúng đoạn positional embedding theo độ dài
                embed_pos = embed_pos[past_key_values_length : inputs_embeds.shape[1] + past_key_values_length, :]
        else:
            # Không dùng cache: chỉ lấy đúng số bước đầu từ positional
            embed_pos = embed_pos[: inputs_embeds.shape[1], :]

        # Cộng positional embedding vào input embedding
        hidden_states = inputs_embeds + embed_pos

        # Áp dụng dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # Chuẩn bị buffer lưu trạng thái nếu cần
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None











class Khaanh(KhaanhPreTrainedModel):
    """
    Mô hình MiniCPM-O kết hợp nhiều mô-đun xử lý đa phương thức (text, hình ảnh, âm thanh, TTS).

    Thừa kế:
        MiniCPMOPreTrainedModel: lớp nền hỗ trợ cấu hình và checkpoint cho mô hình.

    Các mô-đun chính được khởi tạo:
        - LLM (Qwen2ForCausalLM)
        - Vision encoder + resampler
        - Audio encoder + projection
        - TTS module
        - Processor (AutoProcessor)
    """

    def __init__(self, config: KhaanhConfig) -> None:
        """
        Hàm khởi tạo mô hình MiniCPM-O.

        Args:
            config (MiniCPMOConfig): Cấu hình của mô hình, bao gồm các cờ khởi tạo vision/audio/tts.
        """
        super().__init__(config)

        # Mô hình ngôn ngữ chính, dựa trên Qwen2
        self.llm = Qwen2ForCausalLM(config)

        # Gắn patch cho hàm chuẩn bị input khi sinh văn bản (generation)
        self.llm.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, self.llm)

        # Kích thước không gian embedding dùng chung
        self.embed_dim: int = self.llm.config.hidden_size

        # Khởi tạo mô-đun thị giác (nếu được cấu hình bật)
        if self.config.init_vision:
            self.vpm = self.init_vision_module()
            self.vision_dim: int = self.vpm.embed_dim
            self.resampler = self.init_resampler(
                target_dim=self.embed_dim,
                input_dim=self.vision_dim
            )

        # Khởi tạo mô-đun âm thanh (nếu được cấu hình bật)
        if self.config.init_audio:
            self.apm = self.init_audio_module()
            audio_output_dim: int = int(self.apm.config.encoder_ffn_dim // 4)
            # Pooling temporal đầu ra âm thanh (giảm chiều thời gian)
            self.audio_avg_pooler = nn.AvgPool1d(
                kernel_size=self.config.audio_pool_step,
                stride=self.config.audio_pool_step
            )
            # Dựng tầng chiếu (projection layer) để ánh xạ về embed_dim
            self.audio_projection_layer = MultiModalProjector(
                in_dim=audio_output_dim,
                out_dim=self.embed_dim
            )
            self.audio_encoder_layer: int = -1  # chọn layer cuối (mặc định)

        # Khởi tạo mô-đun tổng hợp giọng nói (TTS) nếu được bật
        if self.config.init_tts:
            assert _tts_deps, "Vui lòng đảm bảo đã cài vector_quantize_pytorch và vocos."
            self.tts = self.init_tts_module()

        # Khởi tạo processor để xử lý input (tokenizer, image/audio processor)
        self.processor = AutoProcessor.from_pretrained(
            self.config._name_or_path,
            trust_remote_code=True
        )

        # Token đánh dấu kết thúc chuỗi khi sinh văn bản
        self.terminators = ["<|im_end|>", "<|endoftext|>"]

        # Template mặc định để sinh văn bản cho TTS từ lịch sử hội thoại
        self.default_tts_chat_template: str = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>' }}"
            "{% endif %}"
        )

        # Dùng để ép mô hình không dừng sinh dù gặp token kết thúc
        self.force_no_stop: bool = False

        # Thiết lập lại trạng thái session (dành cho API streaming)
        self.reset_session()

    def reset_session(self) -> None:
        """
        Đặt lại toàn bộ trạng thái của phiên làm việc hiện tại.

        Mục đích:
            - Dùng khi bắt đầu một hội thoại mới hoặc khởi tạo lại pipeline sinh đa phương thức.
            - Xóa cache của LLM và mô-đun âm thanh, reset trạng thái hội thoại người dùng.

        Biến được đặt lại:
            - session_id: ID phiên làm việc hiện tại (None nếu chưa khởi tạo).
            - new_user_msg: Cờ đánh dấu có tin nhắn mới từ người dùng.
            - llm_generated: Cờ đánh dấu LLM đã sinh phản hồi chưa.
            - llm_generate_completed: Cờ đánh dấu LLM đã sinh phản hồi xong chưa.
            - llm_past_key_values: Cache attention của LLM từ các bước trước.
            - audio_past_key_values: Cache attention của mô-đun âm thanh (APM).
        """
        self.session_id: Optional[str] = None                   # ID phiên hiện tại (có thể dùng để định danh)
        self.new_user_msg: bool = True                          # Có tin nhắn mới từ người dùng không
        self.llm_generated: bool = False                        # LLM đã sinh dữ liệu?
        self.llm_generate_completed: bool = False               # LLM đã sinh xong chưa?
        self.llm_past_key_values: Optional[Any] = None          # Cache attention của LLM (giúp tăng tốc)
        self.audio_past_key_values: Optional[Any] = None        # Cache attention của mô-đun âm thanh (APM)

    def init_tts(
            self,
            tts_text_tokenizer_path: Optional[str] = None,
            vocos_ckpt_path: Optional[str] = None,
        ) -> None:
        """
        Khởi tạo hệ thống TTS (Text-to-Speech), gồm:
        1. Tokenizer để mã hóa văn bản đầu vào.
        2. Vocos vocoder để biến đặc trưng âm thanh thành waveform.

        Ưu tiên tải từ local, nếu không có thì tự động tải từ HuggingFace Hub.

        Args:
            tts_text_tokenizer_path (Optional[str]): Đường dẫn tới tokenizer của ChatTTS.
            vocos_ckpt_path (Optional[str]): Đường dẫn tới checkpoint của Vocos.
        """
        from .processing_khaanh import ChatTTSProcessor 

        # Đường dẫn tokenizer (ưu tiên local, fallback sang HF)
        if tts_text_tokenizer_path is None:
            tts_text_tokenizer_path = os.path.join(self.config._name_or_path, "assets/chattts_tokenizer")
        if not os.path.exists(tts_text_tokenizer_path):
            tts_text_tokenizer_path = "chwenjun225/chattts_tokenizer"
        
        # Tải tokenizer và khởi tạo processor
        tts_text_tokenizer = BertTokenizerFast.from_pretrained(tts_text_tokenizer_path)
        self.tts_processor = ChatTTSProcessor(text_tokenizer=tts_text_tokenizer)

        # Đường dẫn checkpoint của vocoder ( ưu tiên local, fallback sang HF )
        if vocos_ckpt_path is None:
            vocos_ckpt_path = os.path.join( self.config._name_or_path, "assets/Vocos.pt" ) 
        if not os.path.exists(vocos_ckpt_path):
            vocos_ckpt_path = hf_hub_download(
                repo_id="chwenjun225/-o-2_6",
                subfolder="assets",
                filename="Vocos.pt"
            )

        # Đảm bảo checkpoint tồn tại trước khi khởi tạo vocoder
        assert os.path.exists(vocos_ckpt_path), "Không tìm thấy file Vocos.pt"
        self.vocos = self.initialize_vocos(vocos_ckpt_path)

    def initialize_vocos(self, ckpt_path: str) -> Vocos:
        """
        Khởi tạo mô hình Vocos từ checkpoint.

        Mô hình Vocos bao gồm:
            - feature_extractor: Trích xuất đặc trưng Mel từ waveform.
            - backbone: Mạng chính học biểu diễn từ Mel.
            - head: ISTFT head để tái tạo waveform.

        Args:
            ckpt_path (str): Đường dẫn tới file checkpoint (.pt) đã huấn luyện của Vocos.

        Returns:
            Vocos: Mô hình Vocos đã nạp trọng số và chuyển sang chế độ eval.
        """
        # Khởi tạo bộ trích đặc trưng Mel Spectrogram
        feature_extractor = instantiate_class(
            args=(),
            init={
                "class_path": "vocos.feature_extractors.MelSpectrogramFeatures",
                "init_args": {
                    "sample_rate": 24000,
                    "n_fft": 1024,
                    "hop_length": 256,
                    "n_mels": 100
                },
            },
        )
        # Khởi tạo mạng backbone chính của Vocos
        backbone = instantiate_class(
            args=(),
            init={
                "class_path": "vocos.models.VocosBackbone",
                "init_args": {
                    "input_channels": 100,
                    "dim": 512,
                    "intermediate_dim": 1536,
                    "num_layers": 8
                },
            },
        )
        # Khởi tạo phần đầu ra (head) dùng để tái tạo waveform từ đặc trưng
        head = instantiate_class(
            args=(),
            init={
                "class_path": "vocos.heads.ISTFTHead",
                "init_args": {
                    "dim": 512,
                    "n_fft": 1024,
                    "hop_length": 256
                }
            },
        )
        # Gộp cá thành phần thành mô hình Vocos 
        vocos = Vocos(feature_extractor, backbone, head).to(self.device).eval().to(torch.float32)
        
        return vocos 

    def init_vision_module(self) -> SiglipVisionTransformer:
        """
        Khởi tạo mô-đun thị giác (Vision Transformer) cho mô hình MiniCPM-O.

        - Thiết lập cơ chế attention phù hợp (FlashAttention2 hoặc eager).
        - Tùy chọn loại bỏ lớp cuối encoder nếu được cấu hình.
        - Gán thêm thuộc tính embed_dim và patch_size cho tiện sử dụng.

        Returns:
            SiglipVisionTransformer: Mô hình Vision Transformer đã khởi tạo sẵn.
        """

        # Thiết lập cơ chế attention cho Vision Transformer
        if self.config._attn_implementation == "flash_attention_2":
            self.config.vision_config._attn_implementation = "flash_attention_2"
        else:
            self.config.vision_config._attn_implementation = "eager"

        # Khởi tạo mô hình SigLIP Vision Transformer từ config
        model = SiglipVisionTransformer(self.config.vision_config)

        # Nếu được yêu cầu, loại bỏ lớp encoder cuối cùng (giảm độ sâu)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]
            
        # Gán embed_dim và patch_size như thuộc tính trực tiếp để tiện truy cập
        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)

        return model
    
    def init_resampler(self, embed_dim: int, vision_dim: int) -> Resampler:
        """
        Khởi tạo mô-đun Perceiver Resampler để nén thông tin ảnh đầu vào.

        Args:
            embed_dim (int): Kích thước không gian embedding của mô hình ngôn ngữ (LLM).
            vision_dim (int): Kích thước đầu ra từ Vision Transformer (SigLIP).

        Returns:
            Resampler: Mô-đun resampler giúp chuyển đổi thông tin ảnh sang dạng phù hợp cho LLM.
        """
        return Resampler(
            num_queries=self.config.query_num,        # Số lượng query token output của resampler
            embed_dim=embed_dim,                      # Độ dài vector embedding mục tiêu
            num_heads=embed_dim // 128,               # Số lượng attention heads, chia theo chuẩn FlashAttention
            kv_dim=vision_dim,                        # Đầu vào: kích thước vector từ mô-đun thị giác
            adaptive=True                             # Kích hoạt chế độ resampler thích ứng (adaptive)
        )

    def init_audio_module(self) -> KhaanhWhisperEncoder:
        """
        Khởi tạo mô-đun xử lý âm thanh (audio encoder) cho mô hình MiniCPM-O.

        Returns:
            MiniCPMWhisperEncoder: Mô hình mã hóa âm thanh, thường là encoder giống Whisper.
        """
        # Tạo encoder âm thanh từ cấu hình audio_config
        model = KhaanhWhisperEncoder(self.config.audio_config)
        return model











# Copy and modified from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
# Modified from https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/modeling_minicpmo.py#L3207
def prepare_inputs_for_generation(
    input_ids: Tensor,
    lm_head_weight: Tensor,
    past_key_values: Optional[Union[Cache, StaticCache, tuple]] = None,
    attention_mask: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    cache_position: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    use_cache: bool = True,
    get_max_length_func: Optional[Callable[[], int]] = None,
) -> Dict[str, Any]:
    """
    Hàm chuẩn bị input cho mô hình sinh văn bản (generation) mà không phụ thuộc vào self/model.

    Args:
        input_ids (Tensor): Token ID đầu vào, shape [batch_size, seq_len].
        lm_head_weight (Tensor): Trọng số của lm_head, dùng để lấy dtype cho attention mask.
        past_key_values (Union[Cache, StaticCache, tuple], optional): Cache từ bước trước.
        attention_mask (Tensor, optional): Mặt nạ attention, 1 cho token hợp lệ, 0 cho padding.
        inputs_embeds (Tensor, optional): Embedding thay thế input_ids, thường chỉ dùng ở bước đầu tiên.
        cache_position (Tensor, optional): Vị trí trong cache, dùng cho attention dạng FlashAttention.
        position_ids (Tensor, optional): Vị trí token, sẽ tự tạo nếu không có.
        use_cache (bool): Có dùng cache trong quá trình sinh hay không.
        get_max_length_func (Callable, optional): Hàm lấy độ dài tối đa của cache (chỉ dùng nếu past là StaticCache).

    Returns:
        Dict[str, Any]: Từ điển chứa các tensor đã xử lý sẵn để truyền vào forward của mô hình.
    """

    if past_key_values is not None:
        # Tính độ dài cache hiện tại
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]

        # Cắt input_ids để chỉ giữ token chưa xử lý
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

    # Tạo position_ids nếu chưa có
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]
            position_ids = position_ids.clone(memory_format=torch.contiguous_format)

    # Chọn inputs_embeds hoặc input_ids
    if inputs_embeds is not None and cache_position is not None and cache_position[0] == 0:
        model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
    else:
        model_inputs = {
            "input_ids": input_ids.clone(memory_format=torch.contiguous_format),
            "inputs_embeds": None
        }

    # Nếu dùng StaticCache, chuẩn bị attention mask 4D đặc biệt
    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            device = model_inputs["inputs_embeds"].device
        else:
            batch_size, sequence_length = model_inputs["input_ids"].shape
            device = model_inputs["input_ids"].device

        dtype = lm_head_weight.dtype
        min_dtype = torch.finfo(dtype).min

        assert get_max_length_func is not None, "Hàm get_max_length_func phải được cung cấp khi dùng StaticCache."

        attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask=attention_mask,
            sequence_length=sequence_length,
            target_length=get_max_length_func(),
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=batch_size,
        )

    # Tổng hợp các input đầu ra
    model_inputs.update({
        "position_ids": position_ids,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "attention_mask": attention_mask,
    })

    return model_inputs














































































