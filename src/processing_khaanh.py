"""Processor class for Khaanh."""

import math
import re
from typing import List
from typing import Tuple
from typing import Literal
from typing import Optional
from typing import Union
from typing import Any

import numpy as np
import torch
import torchaudio
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput
from transformers.tokenization_utils_base import TextInput
from transformers.utils import TensorType

from .image_processing_khaanh import KhaanhBatchFeature

class KhaanhProcessor(ProcessorMixin):
    r"""
    Bộ xử lý tổng hợp cho mô hình Khaanh, tích hợp:
    - Tokenizer (`LlamaTokenizerWrapper`)
    - Image processor (`KhaanhImageProcessor`)
    - Audio feature extractor (`WhisperFeatureExtractor`)

    Bộ xử lý này có khả năng nhận đầu vào đa modal (text + image + audio) và
    biến đổi chúng thành các tensor hoặc cấu trúc đầu vào phù hợp cho mô hình Khaanh.

    Args:
        image_processor (KhaanhImageProcessor, optional): Bộ xử lý ảnh.
        feature_extractor (WhisperFeatureExtractor, optional): Bộ trích đặc trưng âm thanh.
        tokenizer (LlamaTokenizerWrapper, optional): Tokenizer cho văn bản.
    """

    attributes = ["image_processor", "feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
            self,
            image_processor: Optional[Any] = None,
            feature_extractor: Optional[Any] = None,
            tokenizer: Optional[Any] = None
        ):
        # Gọi constructor lớp cha ProcessorMixin để thiết lập các thuộc tính cơ bản
        super().__init__(image_processor, feature_extractor, tokenizer)

        # Gán version từ image_processor để dùng cho theo dõi phiên bản hoặc logic điều kiện
        self.version = image_processor.version

    def __call__(
            self,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]], # hỗ trợ text, danh sách, hoặc token hóa sẵn
            images: Optional[ImageInput] = None,
            audios: Optional[Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]]] = None,
            audio_parts: Optional[List[List[int]]] = None,
            max_length: Optional[int] = None,
            do_pad: Optional[bool] = True,
            max_slice_nums: Optional[int] = None,
            use_image_id: bool = True,
            chunk_input: bool = False,
            return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
            sampling_rate: Optional[int] = 16000,
            **kwargs,
        ) -> KhaanhBatchFeature:
        """
        Xử lý đầu vào đa modal (văn bản + ảnh + âm thanh), trả về đối tượng `KhaanhBatchFeature`
        chứa tất cả thông tin được mã hóa sẵn để nạp vào mô hình MiniCPM-o.

        Args:
            text (Union[str, List[str]]): Câu đầu vào (có thể có placeholder như <image>, <audio>).
            images (ImageInput, optional): Ảnh đầu vào. Có thể là một ảnh, danh sách ảnh hoặc batch ảnh.
            audios (Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]], optional): Âm thanh dạng waveform.
            audio_parts (List[List[int]], optional): Thông tin phân đoạn của audio nếu cần gộp nhiều đoạn liên tiếp.
            max_length (int, optional): Độ dài tối đa của input token.
            do_pad (bool): Có pad ảnh về cùng kích thước trong batch không. Mặc định: True.
            max_slice_nums (int, optional): Số lát tối đa khi chia ảnh thành patches.
            use_image_id (bool): Có gán ID ảnh vào placeholder ảnh không.
            chunk_input (bool): Có chia nhỏ audio thành từng chunk trong chế độ streaming không.
            return_tensors (str or TensorType, optional): Loại tensor đầu ra ('pt', 'np', ...).
            sampling_rate (int, optional): Tần số lấy mẫu của audio, thường là 16000.
            **kwargs: Các đối số khác truyền xuống tokenizer hoặc image/audio processor.

        Returns:
            KhaanhBatchFeature: Gói dữ liệu đầu vào đã mã hóa, bao gồm:
                - input_ids
                - attention_mask
                - pixel_values
                - audio_features
                - image_bounds, audio_bounds, spk_bounds
                - image_sizes, tgt_sizes (kích thước ảnh và patch)
        """
        # Xử lý ảnh nếu có 
        if images is not None:
            image_inputs = self.image_processor(
                images, do_pad=do_pad, max_slice_nums=max_slice_nums, return_tensors=return_tensors
            )
        else:
            image_inputs = None

        # Xử lý âm thanh nếu có 
        if audios is not None:
            audio_features, audio_feature_lens, audio_phs = self.audio_feature_extract(
                audios, audio_parts, chunk_input, sampling_rate
            )
        else:
            audio_features, audio_feature_lens, audio_phs = [], [], []

        # Kết hợp văn bản, hình ảnh và audio placeholders vào văn bản
        model_inputs = self._convert_omni_to_inputs(
            image_inputs=image_inputs,
            audio_phs=audio_phs,
            texts=text,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            max_length=max_length,
            **kwargs,
        )

        # Thêm feature âm thanh
        model_inputs["audio_features"] = audio_features
        model_inputs["audio_feature_lens"] = audio_feature_lens

        return KhaanhBatchFeature(data={**model_inputs})

    def get_audio_placeholder(
            self,
            audio_lens: Union[int, float],
            chunk_input: bool,
            chunk_length: Union[int, float],
        ) -> str:
        """
        Sinh chuỗi placeholder biểu diễn đặc trưng âm thanh cho input audio, được sử dụng để định vị trí
        dữ liệu âm thanh trong mô hình đa modal. Hàm trả về một chuỗi văn bản với
        token <unk> lặp lại tương ứng với số lượng embedding vector đầu ra từ feature extractor.

        Args:
            audio_lens (int | float): Độ dài tín hiệu âm thanh (tính theo số mẫu - sample).
            chunk_input (bool): Nếu True, dữ liệu âm thanh sẽ được chia thành nhiều đoạn nhỏ (chunk).
            chunk_length (int | float): Chiều dài mỗi chunk (tính bằng giây). Áp dụng nếu `chunk_input=True`.

        Returns:
            str: Chuỗi placeholder biểu diễn phần âm thanh, bao gồm các token dạng:
                [AUDIO_START] + "<unk>" * n + [AUDIO_END]
        """
        pool_step = 2  # bước pooling hai lần (ví dụ như trong CNN hoặc encoder)
            
        # Tính chiều dài đặc trưng sau bước trích xuất fbank (dựa trên hop_length)
        feature_lens = math.ceil(audio_lens / self.feature_extractor.hop_length)

        # Qua mỗi tầng pooling 1D stride 2 => giảm 1/2 chiều dài
        feature_lens = (feature_lens - 1) // 2 + 1
        output_lens = (feature_lens - pool_step) // pool_step + 1
    
        if chunk_input:
            # Nếu chia chunk, tính số lượng embedding trong mỗi chunk
            fbank_feat_in_chunk = int(chunk_length * 100)  # mỗi 10ms có 1 frame => 100 frame/giây
            cnn_feat_in_chunk = (fbank_feat_in_chunk - 1) // 2 + 1
            audio_embeds_in_chunk = (cnn_feat_in_chunk - pool_step) // pool_step + 1

            # Tính tổng số chunk cần thiết để bao toàn bộ audio
            num_audio_chunks = (output_lens + audio_embeds_in_chunk - 1) // audio_embeds_in_chunk

            place_holders = ""
            total_unk_len = 0
            for _ in range(num_audio_chunks):
                # Số lượng <unk> trong chunk này không vượt quá phần còn lại
                unk_len = min(audio_embeds_in_chunk, output_lens - total_unk_len)
                place_holders += self.tokenizer.audio_start + "<unk>" * unk_len + self.tokenizer.audio_end
                total_unk_len += unk_len

            audio_placeholder = place_holders
        else:
            # Nếu không chia chunk, placeholder là một khối duy nhất
            audio_placeholder = self.tokenizer.audio_start + "<unk>" * output_lens + self.tokenizer.audio_end

        return audio_placeholder

    def audio_feature_extract(
            self,
            audios: Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]],
            audio_parts: Optional[list] = None,
            chunk_input: Optional[bool] = False,
            sampling_rate: Optional[int] = None,
            chunk_length: Optional[int] = 1,
            **kwargs,
        ) -> Tuple[Union[torch.Tensor, List], List[Union[torch.Tensor, List]], List[List[str]]]:
        """
        Trích xuất đặc trưng âm thanh từ dữ liệu đầu vào đa dạng.

        Args:
            audios (Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]]):
                Danh sách âm thanh đầu vào. Có thể là mảng đơn, danh sách mảng hoặc danh sách lồng nhau.
            audio_parts (Optional[List[List[int]]]):
                Thông tin nhóm các đoạn âm thanh thuộc cùng một phần. Dùng để gộp lại trước khi trích xuất.
            chunk_input (Optional[bool], mặc định=False):
                Nếu True, phân đoạn âm thanh theo chunk_length trước khi tạo placeholder.
            sampling_rate (Optional[int]):
                Tần số lấy mẫu của dữ liệu âm thanh.
            chunk_length (Optional[int], mặc định=1):
                Độ dài đoạn âm thanh (tính bằng giây) khi phân chia thành các chunk.
            **kwargs:
                Các đối số thêm để truyền cho `self.feature_extractor`.

        Returns:
            Tuple:
                - audio_features (Tensor hoặc List): Tensor đặc trưng âm thanh đầu ra, có shape [B, C, T] hoặc empty list nếu không có audio.
                - audio_feature_lens_list (List[Tensor hoặc List]): Danh sách chiều dài đặc trưng từng audio đã trích xuất.
                - audio_ph_list (List[List[str]]): Danh sách placeholder chuỗi biểu diễn đặc trưng âm thanh cho từng audio.
        """
        # Chuẩn hóa định dạng đầu vào về dạng List[List[np.ndarray]]
        if isinstance(audios, np.ndarray):
            audios_list = [[audios]]
        elif isinstance(audios[0], np.ndarray):
            audios_list = [audios]
        else:
            audios_list = audios

        if audio_parts is not None:
            assert len(audio_parts) == len(audios_list)
            for parts, audios in zip(audio_parts, audios_list):
                assert len(parts) == len(audios)

        audio_feature_lens_list = []
        audio_ph_list = []
        audio_features_all = []

        # Tạo placeholder chuỗi cho từng đoạn âm thanh (không phụ thuộc audio_parts)
        for audios in audios_list:
            if audios:
                audio_ph_list.append([
                    self.get_audio_placeholder(len(a), chunk_input, chunk_length) for a in audios
                ])
            else:
                audio_ph_list.append([])

        for idx, audios in enumerate(audios_list):
            # Gộp các đoạn âm thanh thuộc cùng một phần (nếu có thông tin phần)
            if audio_parts is not None:
                audio_part = audio_parts[idx]
                merge_audio = []
                cur_audio = []
                for aid, (part, audio) in enumerate(zip(audio_part, audios)):
                    if aid == 0 or audio_part[aid] == audio_part[aid - 1]:
                        cur_audio.append(audio)
                    else:
                        merge_audio.append(np.hstack(cur_audio))
                        cur_audio = [audio]
                if cur_audio:
                    merge_audio.append(np.hstack(cur_audio))
            else:
                merge_audio = audios

            audio_feature_lens = []

            # Nếu âm thanh dài hơn 30s, chia thành các đoạn nhỏ để xử lý
            final_merge_audio = []
            max_audio_inp_len = 30 * sampling_rate
            for audio in merge_audio:
                if len(audio) <= max_audio_inp_len:
                    final_merge_audio.append(audio)
                else:
                    for i in range(math.ceil(len(audio) / max_audio_inp_len)):
                        final_merge_audio.append(audio[i * max_audio_inp_len: (i + 1) * max_audio_inp_len])

            if audios:
                # Gọi feature_extractor để trích xuất đặc trưng âm thanh
                audio_inputs = self.feature_extractor(
                    final_merge_audio,
                    sampling_rate=sampling_rate,
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                    **kwargs,
                )
                audio_feature = audio_inputs["input_features"]
                actual_lens = audio_inputs["attention_mask"].sum(dim=1)

                for feat, lens in zip(audio_feature, actual_lens):
                    audio_features_all.append(feat[:, :lens])
                    audio_feature_lens.append(lens)

                audio_feature_lens = torch.hstack(audio_feature_lens)
                audio_feature_lens_list.append(audio_feature_lens)
            else:
                audio_feature_lens_list.append([])

        # Ghép các đặc trưng âm thanh lại thành tensor batch (nếu có)
        if audio_features_all:
            audio_features = [i.permute(1, 0) for i in audio_features_all]
            audio_features = torch.nn.utils.rnn.pad_sequence(
                audio_features, batch_first=True, padding_value=0.0
            ).permute(0, 2, 1)
        else:
            audio_features = []

        return audio_features, audio_feature_lens_list, audio_ph_list

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        output_ids = args[0]
        result_text = []
        for result in output_ids:
            result = result[result != 0]
            if result[0] == self.tokenizer.bos_id:
                result = result[1:]
            if result[-1] == self.tokenizer.eos_id:
                result = result[:-1]
            result_text.append(self.tokenizer.decode(result, *args[1:], **kwargs).strip())
        return result_text
        # return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        result = args[0]
        result = result[result != 0]
        if result[0] == self.tokenizer.bos_id:
            result = result[1:]
        if result[-1] == self.tokenizer.eos_id or (
            hasattr(self.tokenizer, "eot_id") and result[-1] == self.tokenizer.eot_id
        ):
            result = result[:-1]
        return self.tokenizer.decode(result, *args[1:], **kwargs).strip()

    def _convert(self, input_str, max_inp_length: Optional[int] = None, **kwargs):
        input_ids = self.tokenizer.encode(input_str, **kwargs)
        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        ## image bound
        start_cond = (input_ids == self.tokenizer.im_start_id) | (input_ids == self.tokenizer.slice_start_id)
        end_cond = (input_ids == self.tokenizer.im_end_id) | (input_ids == self.tokenizer.slice_end_id)

        image_start_idx = torch.where(start_cond)[0]
        image_start_idx += 1
        image_end_idx = torch.where(end_cond)[0]

        valid_image_nums = max(len(image_start_idx), len(image_end_idx))

        image_bounds = torch.hstack(
            [
                image_start_idx[:valid_image_nums].unsqueeze(-1),
                image_end_idx[:valid_image_nums].unsqueeze(-1),
            ]
        )

        ##  audio bound
        audio_start_idx = torch.where(input_ids == self.tokenizer.audio_start_id)[0]
        audio_end_idx = torch.where(input_ids == self.tokenizer.audio_end_id)[0]
        assert len(audio_start_idx) == len(audio_end_idx)
        audio_bounds = torch.hstack([(audio_start_idx + 1).unsqueeze(-1), audio_end_idx.unsqueeze(-1)])

        spk_start_idx = torch.where(input_ids == self.tokenizer.spk_start_id)[0]
        spk_end_idx = torch.where(input_ids == self.tokenizer.spk_end_id)[0]
        assert len(spk_start_idx) == len(spk_end_idx)
        spk_bounds = torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])

        return input_ids, image_bounds, audio_bounds, spk_bounds

    def _convert_omni_to_inputs(
        self,
        images,
        audio_phs,
        texts: Union[str, List[str]],
        truncation=None,
        max_length=None,
        max_slice_nums=None,
        use_image_id=None,
        return_tensors=None,
        **kwargs,
    ):
        if images is None and audio_phs is None:
            model_inputs = self.tokenizer(
                texts, return_tensors=return_tensors, truncation=truncation, max_length=max_length, **kwargs
            )
            return KhaanhBatchFeature(data={**model_inputs})

        image_tag = "(<image>./</image>)"
        image_pattern = "\(<image>./</image>\)"
        audio_tag = "(<audio>./</audio>)"
        audio_pattern = "\(<audio>./</audio>\)"
        split_pattern = f"({image_pattern}|{audio_pattern})"

        if isinstance(texts, str):
            texts = [texts]

        bs = len(texts)
        if images is not None:
            images, image_sizes, tgt_sizes = images["pixel_values"], images["image_sizes"], images["tgt_sizes"]
        else:
            images, image_sizes, tgt_sizes = [[]] * bs, [[]] * bs, [[]] * bs

        input_ids_list = []
        image_bounds_list = []
        audio_bounds_list = []
        spk_bounds_list = []

        for index, text in enumerate(texts):
            text_chunks = re.split(split_pattern, text)

            image_tags = re.findall(image_pattern, text)
            audio_tags = re.findall(audio_pattern, text)

            if image_tags:
                assert images is not None
                assert len(image_tags) == len(image_sizes[index])
            if audio_tags:
                assert audio_phs is not None
                assert len(audio_tags) == len(audio_phs[index])

            image_id = 0
            audio_id = 0
            for i, chunk in enumerate(text_chunks):
                if chunk == image_tag:
                    image_placeholder = self.image_processor.get_slice_image_placeholder(
                        image_sizes[index][image_id], image_id, max_slice_nums, use_image_id
                    )
                    image_id += 1
                    text_chunks[i] = image_placeholder
                elif chunk == audio_tag:
                    audio_placeholder = audio_phs[index][audio_id]
                    audio_id += 1
                    text_chunks[i] = audio_placeholder

            final_text = "".join(text_chunks)
            input_ids, image_bounds, audio_bounds, spk_bounds = self._convert(final_text, max_length, **kwargs)

            input_ids_list.append(input_ids)
            image_bounds_list.append(image_bounds)
            audio_bounds_list.append(audio_bounds)
            spk_bounds_list.append(spk_bounds)

        padded_input_ids, padding_lengths = self.pad(input_ids_list, padding_side="left")
        attention_mask = torch.ones_like(padded_input_ids, dtype=torch.bool)
        for i, length in enumerate(padding_lengths):
            image_bounds_list[i] = image_bounds_list[i] + length
            audio_bounds_list[i] = audio_bounds_list[i] + length
            spk_bounds_list[i] = spk_bounds_list[i] + length
            attention_mask[i, :length] = False

        data = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "pixel_values": images,
            "image_sizes": image_sizes,
            "image_bound": image_bounds_list,
            "tgt_sizes": tgt_sizes,
            "audio_bounds": audio_bounds_list,
            "spk_bounds": spk_bounds_list,
        }

        return data

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + feature_extractor_input_names))

    def pad(self, inputs, max_length=None, padding_value=0, padding_side="left"):
        items = []
        if isinstance(inputs[0], list):
            assert isinstance(inputs[0][0], torch.Tensor)
            for it in inputs:
                for tr in it:
                    items.append(tr)
        else:
            assert isinstance(inputs[0], torch.Tensor)
            items = inputs

        batch_size = len(items)
        shape = items[0].shape
        dim = len(shape)
        assert dim <= 2
        if max_length is None:
            max_length = 0
        max_length = max(max_length, max(item.shape[-1] for item in items))
        min_length = min(item.shape[-1] for item in items)
        dtype = items[0].dtype

        if dim == 0:
            return torch.stack([item for item in items], dim=0), [0]
        elif dim == 1:
            if max_length == min_length:
                return torch.stack([item for item in items], dim=0), [0] * batch_size
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        else:
            tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

        padding_length = []
        for i, item in enumerate(items):
            if dim == 1:
                if padding_side == "left":
                    tensor[i, -len(item) :] = item.clone()
                else:
                    tensor[i, : len(item)] = item.clone()
            elif dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item) :, :] = item.clone()
                else:
                    tensor[i, : len(item), :] = item.clone()
            padding_length.append(tensor.shape[-1] - len(item))

        return tensor, padding_length


class MelSpectrogramFeatures(torch.nn.Module):
    def __init__(
        self,
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding: Literal["center", "same"] = "center",
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: Tensor([num_channels, num_samples])
        """
        return super().__call__(audio)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: Tensor([num_channels, num_samples])
        """
        mel: torch.Tensor = self.mel_spec(audio)
        features = torch.log(torch.clip(mel, min=1e-5))
        return features


class ChatTTSProcessor:
    def __init__(self, text_tokenizer):
        self.audio_processor = MelSpectrogramFeatures()
        self.text_tokenizer = text_tokenizer

    def __call__(self, text_list, audio_list):
        assert len(text_list) == len(audio_list)
        input_ids_varlen = []
        for text in text_list:
            input_ids_ = self.text_tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)  # [1, seq_len]
            input_ids_ = input_ids_.squeeze(0)  # [seq_len]
            input_ids_varlen.append(input_ids_)

        audio_features_varlen = []
        for audio in audio_list:
            assert audio.shape.__len__() == 1  # [seq_len]
            try:
                mel = self.audio_processor(audio)  # [100(num_mel_bins), seq_len_mel]
            except Exception as e:
                raise e
            audio_features_varlen.append(mel)

        return {
            "tts_input_ids_varlen": input_ids_varlen,  # return List[Tensor]
            "tts_input_features_varlen": audio_features_varlen,  # return List[Tensor]
        }
