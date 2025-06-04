import math 
import warnings
from functools import partial
from typing import Optional
from typing import Tuple
from typing import Union 
from typing import Callable
from typing import List 
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn.functional import *
from torch.nn.init import trunc_normal_
from torch.nn.modules.activation import *
from transformers.integrations import is_deepspeed_zero3_enabled

def get_2d_sincos_pos_embed(
        embed_dim: int, 
        image_size: Union[int, Tuple[int, int]]
    ) -> np.ndarray:
    """
    Tạo embedding vị trí 2D (sine-cosine) cho các vị trí trong lưới ảnh.

    Args:
        embed_dim (int): Số chiều của embedding (thường chia hết cho 4).
        image_size (int or Tuple[int, int]): Kích thước ảnh đầu vào.
            - Nếu là int: ảnh vuông (H = W).
            - Nếu là tuple: (height, width).

    Returns:
        np.ndarray: Mảng embedding vị trí có shape [H * W, embed_dim], mỗi hàng ứng với 1 patch.
    """
    # Xử lý kích thước ảnh
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    # Tạo lưới tọa độ: grid_h shape [H], grid_w shape [W]
    grid_h = np.arange(grid_h_size, dtype=np.float32) # trục dọc
    grid_w = np.arange(grid_w_size, dtype=np.float32) # trục ngang

    # Tạo lưới 2D theo thứ tự (width, height), shape sau meshgrid: [2, H, W]
    grid = np.meshgrid(grid_w, grid_w) # trục width đi trước 
    grid = np.stack(grid, axis=0) 

    # Gọi hàm tạo positional embedding theo grid tọa độ 2D 
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return pos_embed 

def get_2d_sincos_pos_embed_from_grid(
        embed_dim: int, 
        grid: np.ndarray
    ) -> np.ndarray:
    """
    Tạo 2D sine-cosine positional embedding từ lưới tọa độ 2D. 
    Hàm này là bước trung gian trong việc xây dựng positional encoding cho ảnh (hoặc patch grid).

    Args:
        embed_dim (int): Kích thước embedding tổng thể. Phải là số chẵn.
        grid (np.ndarray): Mảng tọa độ lưới, shape [2, H, W], trong đó:
            - grid[0] là ma trận tọa độ x (chiều rộng)
            - grid[1] là ma trận tọa độ y (chiều cao)

    Returns: 
        np.ndarray: Embedding vị trí dạng sin-cos, shape [H, W, embed_dim]
    """
    assert embed_dim % 2 == 0, "embed_dim phải là số chẵn để chia đều cho x/y"

    # Sử dụng nửa chiều để mã hóa trục x (chiều ngang)
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0]) # Shape: [H, W, D/2]

    # Nửa còn lại để mã hóa trục y (chiều dọc)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # Shape: [H, W, D/2]

    # Ghép lại theo chiều cuối (chiều embedding) → [H, W, D]
    emb = np.concatenate([emb_h, emb_w], axis=-1)

    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Hàm này chính là bước tạo positional encoding theo chuẩn paper "Attention is All You Need" 
    nhưng mở rộng sang không gian 2D.

    Khi dùng cho ảnh: bạn sẽ gọi nó hai lần — một lần cho trục x, một lần cho trục y — và ghép 
    lại để được embedding 2D.

    Tạo positional embedding 1 chiều theo phương pháp sin-cos (dạng ViT),
    áp dụng cho từng tọa độ trong lưới 2D, trả về tensor embedding 3D.

    Args:
        embed_dim (int): Kích thước embedding đầu ra cho mỗi vị trí (phải chia hết cho 2).
        pos (np.ndarray): Ma trận tọa độ vị trí có shape (H, W), thường là một trục x hoặc y từ meshgrid.

    Returns:
        np.ndarray: Mảng embedding vị trí sin-cos có shape (H, W, embed_dim).
    """
    assert embed_dim % 2, "embed_dim phải là số chẵn để ghép sin/cos" 

    # Tính tần số omega (giống như trong Transformers)
    omega = np.arange(embed_dim // 2, dtype=np.float32) # (D/2,)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10_000 ** omega) # (D/2,)

    # Nhân ngoài pos và omega: mỗi-tọa-độ * mỗi-tần-số -> shape(H, W, D/2)
    # Hàm einsum("hw,d->hwd", ...) là cách gọn gàng để tính outer product giữa lưới tọa độ và tần số
    out = np.einsum("hw,d->hwd", pos, omega) # outer product theo chiều vị trí và tần số 

    # Tính sin và cos cho embedding 
    emb_sin = np.sin(out) # (H, W, D/2)
    emb_cos = np.cos(out) # (H, W, D/2)

    # Ghép sin và cos lại thành embedding cuối cùng: (H, W, D)
    emb = np.concatenate([emb_sin, emb_cos], axis=-1)
    return emb 

class Resampler(nn.Module):
    """
    Resampler thường dùng để rút trích thông tin hình ảnh từ một feature map (như 
    của ViT hoặc CNN) thành một số lượng fixed query token, giống như trong Flamingo, 
    Perceiver, MiniGPT, MiniCPM-o...

    MultiheadAttention là lớp attention kiểu torch (cần bạn định nghĩa riêng hoặc import thủ công).

    `self.proj` được khởi tạo thủ công với `scale 1/sqrt(embed_dim)` để khởi tạo tốt hơn.
    
    Mạng Perceiver-Resampler 2D với một lớp cross-attention, dùng để nén đầu vào không 
    gian (image feature map) về số lượng query cố định.

    Mỗi truy vấn (query) là một vector học được, được ghép với positional embedding 2D.

    Output:
        Tensor có shape (batch_size, num_queries, embed_dim)
    """
    def __init__(
            self,
            num_queries: int,
            embed_dim: int,
            num_heads: int,
            kv_dim: Optional[int] = None,
            norm_layer: Callable[[int], nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            adaptive: bool = False,
            max_size: Tuple[int, int] = (70, 70)
        ) -> None:
        """
        Args:
            num_queries (int): Số lượng query cố định dùng để extract thông tin từ feature đầu vào.
            embed_dim (int): Kích thước vector embedding (chiều sâu).
            num_heads (int): Số lượng đầu attention trong MultiheadAttention.
            kv_dim (int, optional): Kích thước của đầu vào key/value. Nếu khác embed_dim thì sẽ chiếu (project).
            norm_layer (Callable): Lớp normalization dùng cho query và key/value (mặc định: LayerNorm).
            adaptive (bool): Có bật chế độ thích ứng (adaptive) không (dùng khi pos emb phụ thuộc kích thước).
            max_size (Tuple[int, int]): Kích thước tối đa của feature map (chiều cao, rộng) dùng để tạo cache pos emb.
        """
        super().__init__() 

        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive
        self.max_size = max_size

        # Các query học được: [num_queries, embed_dim]
        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))

        # Nếu kv_dim khác embed_dim thì chiếu vào cùng không gian
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        # Lớp attention chính
        self.attn = MultiheadAttention(embed_dim, num_heads)

        # Normalization cho query và key/value
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        # LayerNorm và output projection sau attention 
        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim ** -0.5) * torch.randn(embed_dim, embed_dim))

        # Cache positional embedding 2D sin-cos theo max_size 
        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(self, max_size: Tuple[int, int], device: str = "cpu") -> None:
        """
        Tạo và lưu trữ positional embedding 2D (dạng sin-cos) cho kích thước tối đa đã định.

        Args:
            max_size (Tuple[int, int]): Kích thước tối đa của ảnh (height, width) để tạo embedding vị trí. 
            device (str, optional): Thiết bị dùng để lưu buffer. Mặc định là "cpu".
                                    Nếu sử dụng DeepSpeed ZeRO-3, sẽ tự động chuyển sang "cuda".
        """
        # Nếu đang chạy ở chế độ DeepSpeed ZeRO-3 thì bắt buộc buffer phải ở "cuda"
        if is_deepspeed_zero3_enabled():
            device = "cuda"

        # Tạo position-embedding sin-cos từ numpy và chuyển thành tensor trên thiết bị tương ứng 
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(self.embed_dim, max_size)).float().to(device)
        
        # Lưu pos_embed như buffer (không phải tham số học được), không lưu khi save model (persistent=False)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes: torch.Tensor, device: Union[str, torch.device])-> None:
        """
        Điều chỉnh cache positional embedding 2D (sin-cos) nếu kích thước ảnh đầu vào lớn hơn kích thước đã cache.

        Args:
            tgt_sizes (torch.Tensor): Tensor có shape (B, 2), chứa chiều cao và chiều rộng của từng ảnh (patch-wise).
                                    - tgt_sizes[:, 0] là chiều cao
                                    - tgt_sizes[:, 1] là chiều rộng
            device (str or torch.device): Thiết bị để lưu buffer positional embedding (CPU hoặc GPU).
        """
        # Tìm kích thước lớn nhất theo batch (chiều cao và chiều rộng)
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])

        # Nếu ảnh lớn hơn kích thước pos cache hiện tại thì cập nhật cache 
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            # Cập nhật max_size theo kích thước mới lớn hơn
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            # Tạo lại cache positional embedding tương ứng
            self._set_2d_pos_cache(self.max_size, device)

    def _init_weights(self, m: torch.nn.Module) -> None:
        """
        Hàm khởi tạo trọng số cho các lớp trong mô hình. Được sử dụng khi gọi model.apply(self._init_weights).

        Args:
            m (nn.Module): Một lớp (layer) trong mô hình. Hàm sẽ kiểm tra kiểu lớp và khởi tạo tương ứng.
        """
        if isinstance(m, nn.Linear):
            # Khởi tạo trọng số của Linear bằng truncated normal (std=0.02)
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # Khởi tạo bias bằng 0
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            # LayerNorm: bias = 0, weight = 1 (chuẩn hóa ban đầu)
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor, tgt_sizes: Optional[Tensor] = None) -> Tensor:
        """
        Thực hiện forward pass của Resampler: áp dụng attention với positional embedding và truy vấn học được.

        Args:
            x (Tensor): Tensor đầu vào từ backbone, dạng (B, L, D) — batch, số patch, dimension.
            tgt_sizes (Tensor, optional): Kích thước lưới patch mỗi ảnh, shape (B, 2), tương ứng với (H, W) cho từng ảnh.

        Returns:
            Tensor: Tensor sau attention, shape (B, num_queries, D)
        """
        assert x.shape[0] == tgt_sizes.shape[0], "Batch size của input và tgt_sizes không khớp"
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        # Tổng số patch = H * W cho mỗi ảnh
        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        # Đảm bảo pos_embed đủ lớn với ảnh đầu vào
        self._adjust_pos_cache(tgt_sizes, device=device)

        # Xây dựng key_padding_mask để bỏ qua padding trong attention
        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            # Trích xuất pos_embed phù hợp với kích thước ảnh
            pos = self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype)
            pos_embed.append(pos)
            # Đánh dấu padding phía sau nếu ảnh này ngắn hơn max_patch_len
            key_padding_mask[i, patch_len[i]:] = True

        # Ghép pos_embed theo chiều batch và sắp lại thành (L, B, D)
        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)

        # Dự đoán giá trị key/value và chuẩn hóa
        x = self.kv_proj(x)                     # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)      # L * B * D

        # Lấy truy vấn học được và chuẩn hóa
        q = self.ln_q(self.query)              # Q * D

        # Attention: Q * B * D với x + pos_embed làm key/value
        out = self.attn(
            self._repeat(q, bs),              # Q * B * D
            x + pos_embed,                    # L * B * D
            x,                                # L * B * D
            key_padding_mask=key_padding_mask,
        )[0]                                   # Output: Q * B * D

        # Định dạng lại và chiếu về không gian đầu ra
        x = out.permute(1, 0, 2)               # B * Q * D
        x = self.ln_post(x)
        x = x @ self.proj                      # B * Q * D

        return x 

    def _repeat(self, query: Tensor, N: int) -> Tensor:
        """
        Lặp lại truy vấn học được cho mỗi mẫu trong batch.

        Args:
            query (Tensor): Tensor học được của truy vấn, shape (Q, D)
            N (int): Kích thước batch

        Returns:
            Tensor: Tensor có shape (Q, N, D)
        """
        return query.unsqueeze(1).repeat(1, N, 1)


class MultiheadAttention(nn.MultiheadAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            bias: bool = True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            kdim: Optional[int] = None,
            vdim: Optional[int] = None,
            batch_first: bool = False,
            device: Optional[str] = None,
            dtype: Optional[torch.dtype] = None
        ):
        """
        Phiên bản kế thừa của MultiheadAttention từ PyTorch, với `out_proj` được định nghĩa lại bằng `nn.Linear`.

        Args:
            embed_dim (int): Kích thước embedding đầu vào và đầu ra.
            num_heads (int): Số lượng đầu attention.
            dropout (float, optional): Tỷ lệ dropout áp dụng sau attention. Mặc định là 0.0.
            bias (bool, optional): Có sử dụng bias trong các projection layers hay không.
            add_bias_kv (bool, optional): Có thêm bias cho key và value không. Mặc định False.
            add_zero_attn (bool, optional): Có thêm zero attention vào cuối chuỗi không. Mặc định False.
            kdim (int, optional): Kích thước embedding của key (nếu khác với query).
            vdim (int, optional): Kích thước embedding của value (nếu khác với query).
            batch_first (bool, optional): Nếu True, shape input là (B, L, D). Mặc định False (L, B, D).
            device (str, optional): Thiết bị lưu trữ layer (cpu/gpu).
            dtype (torch.dtype, optional): Kiểu dữ liệu tensor.

        Ghi chú:
            Lớp này định nghĩa lại `self.out_proj` bằng `nn.Linear` để thay thế mặc định trong `nn.MultiheadAttention`,
            nhằm có khả năng kiểm soát tốt hơn về thiết lập layer hoặc cho mục đích compatibility.
        """
        super().__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
            kdim, vdim, batch_first, device, dtype
        )

        # Rewrite out_proj layer，with nn.Linear --- Ghi đè lớp chiếu đầu ra mặc định bằng một Linear layer rõ ràng
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
        ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Tính toán đầu ra attention theo cơ chế multi-head, có thể sử dụng thêm key padding mask và attention mask.

        Args:
            query (Tensor): Tensor query có shape (B, T, E) nếu batch_first=True, hoặc (T, B, E) nếu ngược lại.
            key (Tensor): Tensor key, có shape giống với query.
            value (Tensor): Tensor value, có shape giống với key.
            key_padding_mask (Optional[Tensor], optional): Mask để tránh attention vào các vị trí padding.
                Shape (B, T). Mặc định là None.
            need_weights (bool, optional): Có trả về attention weights hay không. Mặc định là True.
            attn_mask (Optional[Tensor], optional): Mask để chặn attention tới một số vị trí nhất định.
                Thường được sử dụng cho attention nhân quả (causal attention). Mặc định là None.
            average_attn_weights (bool, optional): Có tính trung bình attention weights qua các head không.
                Mặc định là True.
            is_causal (bool, optional): Có áp dụng causal masking (chặn attention về tương lai) hay không.
                Mặc định là False.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Tensor đầu ra sau khi tính attention, có shape 
                (B, T, E) nếu batch_first=True, hoặc (T, B, E) nếu ngược lại. Kèm theo attention weights nếu cần.
        """
        # Xác định xem có nên sử dụng phiên bản "fast path" tối ưu hay không
        why_not_fast_path = ""

        # Fast path không hỗ trợ nếu mask là kiểu float
        if (
            (attn_mask is not None and torch.is_floating_point(attn_mask)) or
            (key_padding_mask is not None and torch.is_floating_point(key_padding_mask))
        ):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        # Kiểm tra xem đầu vào có theo batch (kích thước 3 chiều) hay không
        is_batched = query.dim() == 3

        # Chuẩn hóa định dạng mask về đúng kiểu dữ liệu phù hợp
        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        # Kiểm tra các điều kiện để được dùng fast path
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = (
                f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
            )
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = (
                f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
            )
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time is not supported with NestedTensor input"
        
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        # Nếu không có vấn đề gì, thử dùng fast path
        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # Một số điều kiện bổ sung để đảm bảo tính tương thích
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args): 
                why_not_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
                )
            elif torch.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            # Nếu không có trở ngại, sử dụng kernel attention tối ưu
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type,
                    )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def multi_head_attention_forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            embed_dim_to_check: int,
            num_heads: int,
            in_proj_weight: Optional[Tensor],
            in_proj_bias: Optional[Tensor],
            bias_k: Optional[Tensor],
            bias_v: Optional[Tensor],
            add_zero_attn: bool,
            dropout_p: float,
            out_proj_weight: Tensor,
            out_proj_bias: Optional[Tensor],
            training: bool = True,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            use_separate_proj_weight: bool = False,
            q_proj_weight: Optional[Tensor] = None,
            k_proj_weight: Optional[Tensor] = None,
            v_proj_weight: Optional[Tensor] = None,
            static_k: Optional[Tensor] = None,
            static_v: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
        ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Thực hiện tính toán attention nhiều đầu (multi-head attention) với khả năng hỗ trợ:
        - Sử dụng trọng số chiếu chung hoặc riêng cho query, key, value
        - Mask attention theo vị trí và/hoặc padding
        - Áp dụng attention causal
        - Điều chỉnh dropout theo chế độ huấn luyện

        Tham số:
            query (Tensor): Tensor đầu vào truy vấn, có shape (T, B, E).
            key (Tensor): Tensor key, có shape giống query.
            value (Tensor): Tensor value, có shape giống key.
            embed_dim_to_check (int): Giá trị kiểm tra kích thước embedding, phải bằng với kích thước thật của query.
            num_heads (int): Số lượng đầu attention.
            in_proj_weight (Optional[Tensor]): Trọng số chiếu chung cho QKV nếu không dùng chiếu riêng.
            in_proj_bias (Optional[Tensor]): Bias cho trọng số chiếu chung QKV.
            bias_k (Optional[Tensor]): Bias thêm vào key, nếu có.
            bias_v (Optional[Tensor]): Bias thêm vào value, nếu có.
            add_zero_attn (bool): Có thêm attention vector bằng 0 không.
            dropout_p (float): Tỉ lệ dropout được áp dụng sau attention.
            out_proj_weight (Tensor): Trọng số chiếu đầu ra (out projection).
            out_proj_bias (Optional[Tensor]): Bias cho chiếu đầu ra.
            training (bool, mặc định=True): Có đang ở chế độ huấn luyện không (ảnh hưởng đến dropout).
            key_padding_mask (Optional[Tensor]): Mask để che các vị trí padding trong key. Shape (B, S).
            need_weights (bool, mặc định=True): Trả về trọng số attention hay không.
            attn_mask (Optional[Tensor]): Mask để che các vị trí attention (như trong causal attention).
            use_separate_proj_weight (bool): Nếu True, sử dụng trọng số riêng cho Q/K/V thay vì dùng chung.
            q_proj_weight (Optional[Tensor]): Trọng số chiếu cho query (dùng nếu dùng chiếu riêng).
            k_proj_weight (Optional[Tensor]): Trọng số chiếu cho key (dùng nếu dùng chiếu riêng).
            v_proj_weight (Optional[Tensor]): Trọng số chiếu cho value (dùng nếu dùng chiếu riêng).
            static_k (Optional[Tensor]): Key tĩnh được cung cấp sẵn (thường dùng trong decoding).
            static_v (Optional[Tensor]): Value tĩnh được cung cấp sẵn.
            average_attn_weights (bool): Có lấy trung bình attention qua các head không (mặc định: True).
            is_causal (bool): Có áp dụng attention kiểu causal không (tức không được nhìn về tương lai).

        Trả về:
            Tuple[Tensor, Optional[Tensor]]:
                - Tensor output sau khi áp dụng attention, shape (T, B, E).
                - Attention weights nếu need_weights=True, shape (B, T, S) hoặc None nếu không yêu cầu.
        """
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)

        is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

        # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
        # is batched, run the computation and before returning squeeze the
        # batch dimension so that the output doesn't carry this temporary batch dimension.
        if not is_batched:
            # unsqueeze if the input is unbatched
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        if is_causal and attn_mask is None:
            raise RuntimeError(
                "Need attn_mask if specifying the is_causal hint. "
                "You may use the Transformer module method "
                "`generate_square_subsequent_mask` to create this mask."
            )

        if is_causal and key_padding_mask is None and not need_weights:
            # when we have a kpm or need weights, we need attn_mask
            # Otherwise, we use the is_causal hint go as is_causal
            # indicator to SDPA.
            attn_mask = None
        else:
            attn_mask = _canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )

            if key_padding_mask is not None:
                # We have the attn_mask, and use that to merge kpm into it.
                # Turn off use of is_causal hint, as the merged mask is no
                # longer causal.
                is_causal = False
        
        assert (embed_dim == embed_dim_to_check), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
        else:
            head_dim = embed_dim // num_heads
        
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        
        if use_separate_proj_weight:
            # allow MHA to have different embedding dimensions when separate projection weights are used
            assert (key.shape[:2] == value.shape[:2]), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        else:
            assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

        #
        # compute in-projection
        #
        if not use_separate_proj_weight:
            assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:
            assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
            assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
            assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

        # --- prep attention mask --- 
        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # add bias along batch dimension (currently second)
        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if static_k is None:
            k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert (
                static_k.size(0) == bsz * num_heads
            ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
            assert static_k.size(2) == head_dim, f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
            k = static_k
        if static_v is None:
            v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert (
                static_v.size(0) == bsz * num_heads
            ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
            assert static_v.size(2) == head_dim, f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
            v = static_v

        # add zero attention along batch dimension (now first)
        if add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bsz,
                src_len,
            ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, num_heads, -1, -1)
                .reshape(bsz * num_heads, 1, src_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #

        if need_weights:
            B, Nt, E = q.shape
            q_scaled = q / math.sqrt(E)

            assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"

            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = softmax(attn_output_weights, dim=-1)
            if dropout_p > 0.0:
                attn_output_weights = dropout(attn_output_weights, p=dropout_p)

            attn_output = torch.bmm(attn_output_weights, v)

            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = self.out_proj(attn_output)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)

            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            # attn_mask can be either (L,S) or (N*num_heads, L, S)
            # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
            # in order to match the input for SDPA of (N, num_heads, L, S)
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

            q = q.view(bsz, num_heads, tgt_len, head_dim)
            k = k.view(bsz, num_heads, src_len, head_dim)
            v = v.view(bsz, num_heads, src_len, head_dim)

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

            attn_output = self.out_proj(attn_output)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
            return attn_output, None

def _mha_shape_check(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor],
        num_heads: int,
    ) -> bool:
    """
    Kiểm tra hình dạng (shape) đầu vào của các tensor dùng trong Multi-Head Attention (MHA).

    Tham số:
        query (Tensor): Tensor truy vấn, có thể là 2-D (không batch) hoặc 3-D (có batch).
        key (Tensor): Tensor key, phải cùng chiều với `query`.
        value (Tensor): Tensor value, phải cùng chiều với `key`.
        key_padding_mask (Optional[Tensor]): Mask để loại bỏ padding, 1-D hoặc 2-D tùy theo chế độ batch.
        attn_mask (Optional[Tensor]): Attention mask, 2-D hoặc 3-D.
        num_heads (int): Số lượng đầu attention, dùng để kiểm tra shape khi `attn_mask` là 3-D.

    Trả về:
        bool: Trả về `True` nếu đầu vào là dạng batch (3-D), ngược lại là `False`.

    Gây lỗi:
        AssertionError: Nếu các tensor đầu vào không có shape đúng như yêu cầu theo tiêu chuẩn MHA.
    """
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, (
            "For batched (3-D) `query`, expected `key` and `value` to be 3-D"
            f" but found {key.dim()}-D and {value.dim()}-D tensors respectively"
        )
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, (
                "For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                f" but found {key_padding_mask.dim()}-D tensor instead"
            )
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), (
                "For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                f" but found {attn_mask.dim()}-D tensor instead"
            )
    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, (
            "For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
            f" but found {key.dim()}-D and {value.dim()}-D tensors respectively"
        )

        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, (
                "For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                f" but found {key_padding_mask.dim()}-D tensor instead"
            )

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), (
                "For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                f" but found {attn_mask.dim()}-D tensor instead"
            )
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert (
                    attn_mask.shape == expected_shape
                ), f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}"
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor"
        )

    return is_batched


def _canonical_mask(
        mask: Optional[Tensor],
        mask_name: str,
        other_type: Optional[Any],
        other_name: str,
        target_type: Any,
        check_other: bool = True,
    ) -> Optional[Tensor]:
    """
    Chuyển đổi `mask` (mask attention hoặc mask padding) về dạng chuẩn (canonical),
    có kiểu float (dtype) và giá trị `-inf` tại các vị trí bị mask.

    Tham số:
        mask (Optional[Tensor]): Tensor mask đầu vào, có thể là kiểu `bool` hoặc `float`.
        mask_name (str): Tên của mask để hiển thị trong cảnh báo lỗi (debug).
        other_type (Optional[DType]): Kiểu dtype của mask khác (ví dụ giữa `attn_mask` và `key_padding_mask`),
                                        dùng để kiểm tra sự tương thích dtype.
        other_name (str): Tên của mask kia để sử dụng trong thông báo cảnh báo nếu kiểu không khớp.
        target_type (DType): Dtype mục tiêu muốn chuyển đổi về (thường là `float`).
        check_other (bool): Nếu `True`, kiểm tra sự tương thích giữa `mask` và `other_type`.

    Trả về:
        Optional[Tensor]: Trả về mask ở dạng `float` có các phần tử `-inf` tại vị trí bị mask, hoặc `None` nếu đầu vào là `None`.

    Gây lỗi:
        AssertionError: Nếu `mask` không phải kiểu `bool` hoặc `float`.
    """
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)

        # Kiểm tra xem kiểu dữ liệu của mask có hợp lệ hay không
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(f"Chỉ hỗ trợ kiểu dữ liệu bool và float cho `{mask_name}`")

        # Nếu bật kiểm tra và có loại khác (other_type), kiểm tra sự tương thích kiểu
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Hỗ trợ các mask có kiểu dữ liệu không khớp giữa `{mask_name}` và `{other_name}` "
                    "đã lỗi thời. Hãy dùng cùng kiểu dữ liệu cho cả hai."
                )

        # Nếu mask là bool, chuyển thành float và gán -inf tại vị trí bị mask
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(mask, float("-inf"))
            
    return mask

def _in_projection_packed(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
    ) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.
    
    Args: 
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            proj = linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = linear(q, w_q, b_q)
            kv_proj = linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
        - q': :math:`[Qdims..., Eq]`
        - k': :math:`[Kdims..., Eq]`
        - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
