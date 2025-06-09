import os 
from typing import Union 
from typing import Optional
from typing import Any 
from typing import Dict 

from transformers import PretrainedConfig 
from transformers import Qwen2Config
from transformers import WhisperConfig
from transformers.utils import logging

from .modeling_navit_siglip import SiglipVisionConfig

logger = logging.get_logger(__name__)

class KhaanhSliceConfig(PretrainedConfig):
    """
    Cấu hình cho module xử lý slice ảnh trong mô hình.
    Bao gồm các tham số kích thước patch, số lát ảnh tối đa, 
    và độ phân giải chuẩn dùng để resize ảnh.

    Attributes:
        patch_size (int): Kích thước mỗi patch (ví dụ: 14 x 14 pixel).
        max_slice_nums (int): Số lượng lát ảnh tối đa có thể chia.
        scale_resolution (int): Độ phân giải chuẩn để scale ảnh về trước khi chia lát.
    """
    model_type = "khaanh"

    def __init__(
            self,
            patch_size: int = 14,
            max_slice_nums: int = 9,
            scale_resolution: int = 448,
            **kwargs,
        ):
        """
        Khởi tạo cấu hình slice cho mô hình MiniCPM-o.

        Args:
            patch_size (int, optional): Kích thước mỗi patch. Mặc định là 14.
            max_slice_nums (int, optional): Số lát ảnh tối đa. Mặc định là 9.
            scale_resolution (int, optional): Độ phân giải chuẩn. Mặc định là 448.
            **kwargs: Các tham số cấu hình bổ sung khác.
        """
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs
    ) -> "PretrainedConfig":
        """
        Tải cấu hình từ tên mô hình hoặc đường dẫn đã huấn luyện trước.

        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]): Tên mô hình trên Hugging Face Hub
                hoặc đường dẫn đến thư mục chứa tệp cấu hình.
            **kwargs: Các tham số bổ sung (ví dụ: cache_dir, force_download, revision, v.v.).

        Returns:
            PretrainedConfig: Một đối tượng cấu hình được khởi tạo từ file cấu hình pretrained.
        """
        # Gắn token nếu cầu (dành cho việc tải mô hình private từ 🤗 Hub)
        cls._set_token_in_kwargs(kwargs)

        # Đọc config từ file JSON hoặc remote model hub 
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        # Lấy phần cấu hình cho "slice_config":
        if config_dict.get("model_type") == "minicpmv":
            config_dict = config_dict["slice_config"]

        # Cảnh báo nếu model_type trong file config không trùng với class này
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )
        return cls.from_dict(config_dict, **kwargs)

class ConditionalChatTTSConfig(PretrainedConfig):
    """
    Cấu hình cho mô hình Conditional-Chat-Text-To-Speech.

    Attributes:
        - Kiến trúc mô hình (transformer dimensions, attention, hidden layers, etc)
        - Số lượng token (âm thanh, văn bản)
        - Cấu hình streaming/inference
        - Các tùy chọn cho sampling và loss
    """
    model_type = "conditional_chattts"

    def __init__(
        self,
        llm_dim: int = 2560,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 20,
        max_position_embeddings: int = 4096,
        num_audio_tokens: int = 626,
        num_text_tokens: int = 21178,
        num_mel_bins: int = 100,
        num_vq: int = 4,
        use_speaker_embedding: bool = True,
        use_llm_hidden_state: bool = False,
        spk_emb_token_id: int = 21143,
        num_spk_embs: int = 1,
        audio_bos_token_id: int = 21132,
        text_eos_token_id: int = 21133,
        use_text: bool = True,
        streaming: bool = True,
        streaming_text_chunk_size: int = 10,
        streaming_text_reserved_len: int = 300,
        streaming_audio_chunk_size: int = 50,
        attn_implementation: str = "sdpa",
        use_mlp: bool = True,
        aug_loss_weight: bool = True,
        do_sample: bool = True,
        top_p: float = 0.7,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        **kwargs,
    ):
        """
        Khởi tạo cấu hình cho Conditional ChatTTS.

        Args:
            llm_dim (int): Kích thước ẩn của đầu ra từ LLM (nếu dùng).
            hidden_size (int): Kích thước ẩn của transformer.
            intermediate_size (int): Kích thước tầng FFN.
            num_attention_heads (int): Số lượng đầu attention.
            num_hidden_layers (int): Số tầng transformer.
            max_position_embeddings (int): Chiều dài chuỗi đầu vào tối đa.
            num_audio_tokens (int): Số lượng token trong không gian mã hóa âm thanh.
            num_text_tokens (int): Số lượng token từ tokenizer văn bản.
            num_mel_bins (int): Số lượng bins trong spectrogram đầu ra.
            num_vq (int): Số lượng vector quantizers (VQ) dùng để mã hóa âm thanh.
            use_speaker_embedding (bool): Có sử dụng embedding người nói không.
            use_llm_hidden_state (bool): Có dùng hidden state từ LLM để điều kiện hóa không.
            spk_emb_token_id (int): ID token đại diện cho embedding người nói.
            num_spk_embs (int): Số lượng embedding người nói có thể dùng.
            audio_bos_token_id (int): ID cho token bắt đầu chuỗi âm thanh.
            text_eos_token_id (int): ID cho token kết thúc văn bản.
            use_text (bool): Có sử dụng đầu vào văn bản không (có thể False nếu chỉ dự đoán tiếp âm thanh).
            streaming (bool): Có bật chế độ streaming không.
            streaming_text_chunk_size (int): Kích thước khối văn bản mỗi lần streaming.
            streaming_text_reserved_len (int): Độ dài bộ nhớ giữ lại giữa các chunk văn bản khi streaming.
            streaming_audio_chunk_size (int): Kích thước khối âm thanh khi streaming.
            attn_implementation (str): Kiểu attention sử dụng (vd: "sdpa", "flash").
            use_mlp (bool): Có dùng MLP trong kiến trúc không.
            aug_loss_weight (bool): Có dùng trọng số cho loss bổ sung không.
            do_sample (bool): Có dùng sampling trong quá trình generate không.
            top_p (float): Tham số nucleus sampling (p).
            top_k (int): Tham số top-k sampling.
            repetition_penalty (float): Phạt lặp lại token khi generate.
            **kwargs: Các tham số cấu hình bổ sung khác.
        """
        super().__init__(**kwargs)
        
        # Cấu hình kiến trúc
        self.llm_dim = llm_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings

        # Token & embedding
        self.num_audio_tokens = num_audio_tokens
        self.num_text_tokens = num_text_tokens
        self.num_mel_bins = num_mel_bins
        self.num_vq = num_vq

        # Cấu hình giọng nói và LLM
        self.use_speaker_embedding = use_speaker_embedding
        self.use_llm_hidden_state = use_llm_hidden_state
        self.spk_emb_token_id = spk_emb_token_id
        self.num_spk_embs = num_spk_embs
        self.audio_bos_token_id = audio_bos_token_id
        self.text_eos_token_id = text_eos_token_id

        # Tùy chọn xử lý đầu vào
        self.use_text = use_text

        # Streaming inference
        self.streaming = streaming
        self.streaming_text_chunk_size = streaming_text_chunk_size
        self.streaming_text_reserved_len = streaming_text_reserved_len
        self.streaming_audio_chunk_size = streaming_audio_chunk_size

        # Các kỹ thuật cải tiến
        self.attn_implementation = attn_implementation
        self.use_mlp = use_mlp
        self.aug_loss_weight = aug_loss_weight

        # Tùy chọn sampling
        self.do_sample = do_sample
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

class KhaanhConfig(Qwen2Config):
    """
    Cấu hình tổng hợp cho mô hình Khaanh (Khaanh-Multi-Modal),
    tích hợp các mô đun xử lý văn bản, hình ảnh, âm thanh, và text-to-speech (TTS).

    Kế thừa từ Qwen2Config và mở rộng với các cấu hình phụ:
    - Vision (SigLIP)
    - Audio (Whisper)
    - TTS (ConditionalChatTTS)
    - Slice (KhaanhSliceConfig)
    """
    model_type = "khaanh"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Cấu hình mặc định cho mô hình vision SigLIP nếu không cung cấp
    default_vision_config = {
        "hidden_size": 1152,
        "image_size": 980,
        "intermediate_size": 4304,
        "model_type": "siglip",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }

    def __init__(
            self,
            use_cache: bool = True,
            query_num: int = 64,
            image_size: int = 448,
            drop_vision_last_layer: bool = True,
            batch_vision_input: bool = True,
            slice_config: Optional[Union[Dict[str, Any], KhaanhSliceConfig]] = None,
            vision_config: Optional[Union[Dict[str, Any], SiglipVisionConfig]] = None,
            audio_config: Optional[Union[Dict[str, Any], WhisperConfig]] = None,
            tts_config: Optional[Union[Dict[str, Any], ConditionalChatTTSConfig]] = None,
            use_image_id: bool = True,
            vision_batch_size: int = 16,
            audio_pool_step: int = 2,
            audio_chunk_length: float = 1.0,
            stream_input: bool = False,
            init_vision: bool = True,
            init_audio: bool = True,
            init_tts: bool = True,
            **kwargs,
        ):
        """
        Khai báo cấu trúc và hành vi cho các mô đun phụ của mô hình.

        Args:
            use_cache (bool): Có sử dụng bộ nhớ attention trong inference không.
            query_num (int): Số lượng query token dùng để nạp thông tin ảnh vào LLM.
            image_size (int): Kích thước ảnh (vuông).
            drop_vision_last_layer (bool): Có loại bỏ lớp cuối của encoder ảnh không.
            batch_vision_input (bool): Có xử lý batch ảnh hay từng ảnh lẻ.
            slice_config (Union[dict, MiniCPMVSliceConfig], optional): Cấu hình chia lát ảnh.
            vision_config (Union[dict, SiglipVisionConfig], optional): Cấu hình encoder ảnh (SigLIP).
            audio_config (Union[dict, WhisperConfig], optional): Cấu hình cho mô hình âm thanh (Whisper).
            tts_config (Union[dict, ConditionalChatTTSConfig], optional): Cấu hình cho mô hình TTS.
            use_image_id (bool): Có sử dụng ID ảnh trong placeholder không.
            vision_batch_size (int): Batch size cho xử lý ảnh.
            audio_pool_step (int): Số bước stride trong pooling encoder âm thanh.
            audio_chunk_length (float): Độ dài mỗi đoạn âm thanh (tính bằng giây).
            stream_input (bool): Có bật chế độ streaming input không.
            init_vision (bool): Khởi tạo encoder ảnh khi load không.
            init_audio (bool): Khởi tạo encoder âm thanh khi load không.
            init_tts (bool): Khởi tạo TTS decoder khi load không.
            **kwargs: Các tham số khác kế thừa từ Qwen2Config.
        """
        # Các tham số tổng quát
        self.use_cache = use_cache
        self.query_num = query_num
        self.image_size = image_size
        self.drop_vision_last_layer = drop_vision_last_layer
        self.batch_vision_input = batch_vision_input
        self.use_image_id = use_image_id
        self.vision_batch_size = vision_batch_size
        self.audio_pool_step = audio_pool_step
        self.audio_chunk_length = audio_chunk_length
        self.stream_input = stream_input
        self.init_vision = init_vision
        self.init_audio = init_audio
        self.init_tts = init_tts

        # Slice config
        if slice_config is None:
            self.slice_config = KhaanhSliceConfig(max_slice_nums=1)
        else:
            self.slice_config = (
                slice_config if isinstance(slice_config, KhaanhSliceConfig)
                else KhaanhSliceConfig(**slice_config)
            )
        self.slice_mode = True  # Luôn bật chia lát ảnh nếu dùng Khaanh

        # Vision config (dùng SigLIP)
        if vision_config is None:
            self.vision_config = SiglipVisionConfig(**self.default_vision_config)
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = SiglipVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        # Audio config (dùng Whisper)
        if audio_config is None:
            self.audio_config = WhisperConfig()
        elif isinstance(audio_config, dict):
            self.audio_config = WhisperConfig(**audio_config)
        else:
            self.audio_config = audio_config

        # TTS config
        if tts_config is None:
            self.tts_config = ConditionalChatTTSConfig()
        elif isinstance(tts_config, dict):
            self.tts_config = ConditionalChatTTSConfig(**tts_config)
        else:
            self.tts_config = tts_config

        # Lưu patch_size để các module con dễ truy cập
        self.patch_size = self.vision_config.patch_size

        # Gọi constructor cha (Qwen2)
        super().__init__(**kwargs)
