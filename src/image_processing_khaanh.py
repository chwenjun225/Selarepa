# Code reference from https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/image_processing_minicpmv.py
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from typing import Union

import numpy as np
import PIL
import PIL.Image
import PIL.ImageSequence
import torch
from transformers import AutoImageProcessor
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import to_channel_dimension_format
from transformers.image_utils import ChannelDimension
from transformers.image_utils import infer_channel_dimension_format
from transformers.image_utils import is_torch_tensor
from transformers.image_utils import to_numpy_array
from transformers.image_utils import valid_images
from transformers.utils import is_torch_device
from transformers.utils import is_torch_dtype
from transformers.utils import requires_backends
from transformers.utils import TensorType


def recursive_converter(converter, value):
    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value += [recursive_converter(converter, v)]
        return new_value
    else:
        return converter(value)


class KhaanhBatchFeature(BatchFeature):
    r"""
    Class mở rộng từ `BatchFeature`, nhằm hỗ trợ xử lý batch với các 
    hình ảnh có kích thước khác nhau.
    
    Attributes:
        data (Optional[Dict[str, Any]]): Dữ liệu batch đầu vào.
        tensor_type (Union[None, str, TensorType]): Loại tensor muốn 
            chuyển đổi dữ liệu sang (ví dụ: "pt", "tf", v.v.).
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None, tensor_type: Union[None, str, TensorType] = None):
        """
        Khởi tạo `KhaanhBatchFeature`, đồng thời chuyển đổi dữ liệu 
        sang tensor nếu chỉ định.

        Args:
            data (Optional[Dict[str, Any]]): Dữ liệu batch đầu vào.
            tensor_type (Union[None, str, TensorType], optional): Loại 
                tensor để chuyển dữ liệu sang.
        """
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def convert_to_tensors(
            self, 
            tensor_type: Optional[Union[str, TensorType]] = None
        )-> "KhaanhBatchFeature":
        """
        Chuyển đổi dữ liệu trong batch sang tensor tương ứng với `tensor_type`.

        Args:
            tensor_type (Optional[Union[str, TensorType]]): Kiểu tensor 
                để chuyển đổi ("pt", "tf", "np", v.v.).

        Returns:
            KhaanhBatchFeature: Đối tượng hiện tại đã chuyển đổi dữ liệu thành tensor.

        Raises:
            ValueError: Nếu không thể tạo tensor do các chuỗi có độ dài khác nhau.
        """
        if tensor_type is None:
            return self
        # Lấy hai hàm kiểm tra và chuyển đổi tensor tương ứng với loại được chỉ định 
        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)
        # Hàm chuyển đổi đệ quy từng phần tử 
        def converter(value):
            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)
                    return tensor
            except:  # noqa E722
                if key == "overflowing_values":
                    raise ValueError(
                        "Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )
        # Lặp qua từng key trong batch và áp dụng chuyển đổi đệ quy 
        for key, value in self.items():
            self[key] = recursive_converter(converter, value)
        return self

    def to(self, *args, **kwargs) -> "KhaanhBatchFeature":
        """
        Chuyển tất cả các tensor trong `KhaanhBatchFeature` sang thiết bị cuda/cpu 
        hoặc chuyển thành kiểu dữ liệu khác (dtype).

        Phương thức này sử dụng tương tự như `.to()` trong Pytorch để chuyển tensor 
        sang GPU/CPU hoặc chuyển kiểu dữ liệu (ví dụ: float32 -> float16). Nó áp dụng 
        một cách đệ quy trên toàn bộ các phần từ trong `KhaanhBatchFeature`.

        Args:
            *args: Tham số chuyển tiếp tới `torch.Tensor.to()` như device (ví dụ: 
                "cuda", "cpu") hoặc dtype (ví dụ: `torch.float16`
            **kwargs: Các tham số bổ xung như `device=...`, `dtype=...`
        
        Returns:
            KhaanhBatchFeature: Đối tượng hiện tại với tất cả tensor đã được chuyển đổi 
                thiết bị hoặc kiểu dữ liệu.
        """
        # Đảm bảo Pytorch đã được cài đặt 
        requires_backends(self, ["torch"])
        import torch 

        # Hàm chuyển tensor 
        def cast_tensor(v: Any) -> Any:
            # check if v is a floating point -- Nếu tensor là số thực (float), cho phép 
            #   thay đổi cả dtype và device 
            if torch.is_floating_point(v):
                # cast and send to device
                return v.to(*args, **kwargs)
            # Nếu không phải float tensor, thì chuyển đổi thiết bị tính toán CUDA/CPU 
            elif device is not None:
                return v.to(device=device)
            else:
                return v
        
        new_data = {}
        device = kwargs.get("device")

        # Check if the args are a device or a dtype 
        if device is None and len(args) > 0:
            # device should be always the first argument
            arg = args[0]
            if is_torch_dtype(arg):
                # If the first argument is a dtype --> Pass 
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                # it's something else
                raise ValueError(
                    f"Attempting to cast a BatchFeature to type" 
                    " {str(arg)}. This is not supported.")
        # We cast only floating point tensors to avoid issues with tokenizers 
        #   casting `LongTensor` to `FloatTensor`
        for k, v in self.items():
            new_data[k] = recursive_converter(cast_tensor, v)
        self.data = new_data
        return self


class KhaanhImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"] 

    def __init__(
            self, 
            max_slice_nums:int=9, 
            scale_resolution:int=448, 
            patch_size:int=14, 
            **kwargs: Dict[str, Any]
        ) -> None:
        """
        Class mở rộng từ `BaseImageProcessor`, để tiền xử lý ảnh 
            tùy chỉnh cho multi-modal-LLM.

        Args:
            max_slice_nums (int): Số lượng lát ảnh (slice) tối đa để xử lý mỗi ảnh.
            scale_resolution (int): Độ phân giải chuẩn hoá của ảnh đầu vào.
            patch_size (int): Kích thước patch dùng để trích xuất đặc trưng từ ảnh.
            **kwargs (Any): Các tham số mở rộng cho cấu hình bổ sung như token đặc biệt,
                            chuẩn hoá, chế độ slice,...
        """

        super().__init__(**kwargs)
        
        # Cấu hình chính cho xử lý ảnh 
        self.max_slice_nums = max_slice_nums # Số lát cắt ảnh tối đa 
        self.scale_resolution = scale_resolution # Độ phân giải resize 
        self.patch_size = patch_size # Kích thước patch trích xuất đặc trưng 
        
        # Các cấu hình đặc biệt lấy từ kwargs 
        self.use_image_id = kwargs.pop("use_image_id", False)
        self.image_feature_size = kwargs.pop("image_feature_size", 64) # Số chiều đặc trưng của ảnh 

        # Các token đặc biệt được sử dụng để mã hóa ảnh 
        self.im_start_token = kwargs.pop("im_start", "<image>")
        self.im_end_token = kwargs.pop("im_end", "</image>")
        self.slice_start_token = kwargs.pop("slice_start", "<slice>")
        self.slice_end_token = kwargs.pop("slice_end", "</slice>")
        self.unk_token = kwargs.pop("unk", "<unk>") # Token <unk> cho patch chưa biết 

        self.im_id_start = kwargs.pop("im_id_start", "<image_id>")
        self.im_id_end = kwargs.pop("im_id_end", "</image_id>")
        self.slice_mode = kwargs.pop("slice_mode", True) # Có xử lý theo lát cắt ảnh không 

        # Chuẩn hóa theo mean và std
        self.mean = np.array(kwargs.pop("norm_mean", [0.5, 0.5, 0.5]))
        self.std = np.array(kwargs.pop("norm_std", [0.5, 0.5, 0.5]))
        
        self.version = kwargs.pop("version", 2.0)
    # TODO: https://chatgpt.com/c/683dad62-5b08-800a-84b1-4aa47a6876b3
    def ensure_divide(self, length: int, patch_size: int) -> int:
        """
        Đảm bảo kích thước `length` chia hết cho `patch_size`.

        Args:
            length (int): Chiều dài hoặc chiều rộng ban đầu.
            patch_size (int): Kích thước của patch (ô ảnh) mà mô hình yêu cầu.

        Returns:
            int: Kích thước đã được làm tròn sao cho chia hết cho `patch_size`, 
                ít nhất bằng `patch_size`.
        """
        # Làm tròn để chia hết cho patch_size, nhưng không nhỏ hơn patch_size
        return max(round(length / patch_size) * patch_size, patch_size)
    
    def find_best_resize( 
            self, 
            original_size: Tuple[int, int], 
            scale_resolution: int, 
            patch_size: int, 
            allow_upscale: bool = False
        ) -> Tuple[int, int]:
        """Adaptive Visual Encoding method proposed by LLaVA-UHD."""
        width, height = original_size
        if (width * height > scale_resolution * scale_resolution) or allow_upscale:
            r = width / height
            height = int(scale_resolution / math.sqrt(r))
            width = int(height * r)
        best_width = self.ensure_divide(width, patch_size)
        best_height = self.ensure_divide(height, patch_size)
        return (best_width, best_height)
    
    def get_refine_size(
            self,
            original_size: Tuple[int, int],
            grid: Tuple[int, int],
            scale_resolution: int,
            patch_size: int,
            allow_upscale: bool = False
        ):
        """
        Tính toán kích thước ảnh đã điều chỉnh (refined size) sao cho:
        - Chia đều thành các ô lưới
        - Phù hợp với độ phân giải và kích thước patch yêu cầu

        Args:
            original_size (Tuple[int, int]): Kích thước ảnh gốc (width, height)
            grid (Tuple[int, int]): Số lượng ô theo chiều ngang và dọc (grid_x, grid_y)
            scale_resolution (int): Độ phân giải mục tiêu để scale
            patch_size (int): Kích thước patch muốn xử lý
            allow_upscale (bool, optional): Có cho phép phóng to ảnh hay không

        Returns:
            Tuple[int, int]: Kích thước mới (refined width, refined height)
        """
        width, height = original_size 
        grid_x, grid_y = grid 

        # Tính kích thước mới sao cho mỗi chiều chia hết cho số ô grid tương ứng 
        refine_width = self.ensure_divide(width, grid_x)
        refine_height = self.ensure_divide(height, grid_y)

        # Tính kích thước mỗi ô lưới (grid cell) sau khi chia đều 
        grid_width = refine_width / grid_x
        grid_height = refine_height / grid_y

        # Tìm kích thước mỗi ô grid tối ưu nhất (gần với scale_resolution, patch_size, etc)
        best_grid_size = self.find_best_resize(
            (grid_width, grid_height), 
            scale_resolution, 
            patch_size, 
            allow_upscale=allow_upscale
        )

        # Nhân ngược lại để ra kích thước tổng (ô * số lượng ô)
        refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)
        return refine_size

    def split_to_patches(
            self, 
            image: PIL.Image.Image, 
            grid: Tuple[int, int]
        ) -> List[List[PIL.Image.Image]]:
        """
        Chia cột thành các ảnh nhỏ (patches) theo lưới (grid) chỉ định.

        Args:
            image (PIL.Image.Image): Ảnh đầu vào cần chia nhỏ 
            grid (Tuple[int, int]): Kích thước lưới chia ảnh, dạng (số cột, số hàng)

        Returns:
            List[List[PIL.Image.Image]]: Danh sách 2 chiều các ảnh nhỏ sau khi cắt.
                Mỗi phần tử là một dòng (hàng) chứa các ảnh patch theo cột.
        """
        patches = []
        width, height = image.size

        # Chiều rộng mỗi patch = tổng chiều rộng chia số cột 
        grid_x = int(width / grid[0])

        # Chiều cao mỗi patch = tổng chiều cao chia số hàng 
        grid_y = int(height / grid[1])
        
        # Duyệt theo chiều dọc của ảnh để chia theo hàng 
        for i in range(0, height, grid_y):
            images = []
            # Duyệt theo chiều ngang trong mỗi hàng để chia theo cột 
            for j in range(0, width, grid_x):

                # Xác định vùng box của patch 
                box = (j, i, j + grid_x, i + grid_y)
                # Cắt patch từ vùng box 
                patch = image.crop(box)
                images.append(patch)
            # Thêm một hàng các patch 
            patches.append(images)

        return patches

    def slice_image(
            self,
            image: PIL.Image.Image,
            max_slice_nums: int = 9,
            scale_resolution: int = 448,
            patch_size: int = 14,
            never_split: bool = False
        ) -> Tuple[PIL.Image.Image, List[List[PIL.Image.Image]], Optional[Tuple[int, int]]]:
        """
        Xử lý ảnh đầu vào bằng cách resize và/hoặc chia lát ảnh (slicing) thành các patch nhỏ tùy theo cấu hình.

        Args:
            image (PIL.Image.Image): Ảnh đầu vào cần xử lý.
            max_slice_nums (int, optional): Số lượng lát ảnh tối đa được chia. Mặc định là 9.
            scale_resolution (int, optional): Độ phân giải chuẩn để resize ảnh. Mặc định là 448.
            patch_size (int, optional): Kích thước patch cơ sở (dùng để đảm bảo chia hết). Mặc định là 14.
            never_split (bool, optional): Nếu đặt True, luôn resize ảnh mà không chia lát. Mặc định là False.

        Returns:
            Tuple:
                - source_image (PIL.Image.Image): Ảnh đã được resize phù hợp để đưa vào mô hình.
                - patches (List[List[PIL.Image.Image]]): Danh sách các patch ảnh (nếu có chia lát, else rỗng).
                - best_grid (Optional[Tuple[int, int]]): Kích thước lưới chia lát tốt nhất (số cột, số hàng), 
                    hoặc None nếu không chia.
        """
        original_size = image.size
        source_image = None
        patches = []

        # Tìm cấu hình chia lưới phù hợp (None nếu không cần chia lát)
        best_grid = self.get_sliced_grid(original_size, max_slice_nums, never_split)

        if best_grid is None:
            # Không cần chia lát, chỉ resize để phù hợp với mô hình 
            best_size = self.find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=True)
            source_image = image.resize(best_size, resample=PIL.Image.Resampling.BICUBIC)
        else:
            # Resize ảnh gốc để chia lát, đảm bảo chia hết cho patch_size 
            best_resize = self.find_best_resize(original_size, scale_resolution, patch_size)
            source_image = image.copy().resize(best_resize, resample=PIL.Image.Resampling.BICUBIC)
            
            # Resize lại ảnh để đảm bảo đúng với kích thước lưới 
            refine_size = self.get_refine_size(
                original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
            )
            refine_image = image.resize(refine_size, resample=PIL.Image.Resampling.BICUBIC)
            
            # Chia ảnh thành các patch theo lưới đã tính 
            patches = self.split_to_patches(refine_image, best_grid)

        return source_image, patches, best_grid

    def get_grid_placeholder(self, grid: Optional[Tuple[int, int]]) -> str:
        """
        Tạo input text dạng token tương đương với ảnh đã được cắt lát, 
            để phục vụ giai đoạn huấn luyện hoặc inference text-only.

        Args:
            grid (Optional[Tuple[int, int]]): Kích thước lưới (số cột, số hàng) chia lát ảnh.
                                            Nếu là None, trả về chuỗi rỗng.

        Returns:
            str: Chuỗi placeholder đại diện cho toàn bộ grid lát ảnh, với mỗi lát là một token pattern.
                Kết quả là nhiều dòng, mỗi dòng chứa `cols` placeholder lát.
        """
        if grid is None: return ""

        # Một placeholder cho một lát ảnh (slice), định dạng như: <slice> <unk>... </slice>
        slice_image_placeholder = (
            self.slice_start_token + \
            self.unk_token * self.image_feature_size + \
            self.slice_end_token
        )

        cols, rows = grid 
        slices = []

        # Tạo từng dòng lát ảnh (dòng = hàng lát), mỗi dòng có 'cols' placeholder lát
        for i in range(rows):
            lines = []
            for j in range(cols):
                lines.append(slice_image_placeholder)
            slices.append("".join(lines)) # Ghép thành một dòng 
        
        slice_placeholder = "\n".join(slices)  # Ghép các dòng thành block placeholder
        return slice_placeholder

    def get_image_id_placeholder(self, idx=0):
        return f"{self.im_id_start}{idx}{self.im_id_end}"
    
    def get_sliced_images(
            self, 
            image: PIL.Image.Image, 
            max_slice_nums: Optional[int] = None
        ) -> List[PIL.Image.Image]:
        """
        Trả về danh sách các ảnh đã được chia lát từ ảnh đầu vào.
        Gồm cả ảnh gốc đã resize (`source_image`) và các patch ảnh (nếu chia lát).

        Args:
            image (PIL.Image.Image): Ảnh đầu vào cần xử lý.
            max_slice_nums (Optional[int]): Số lượng lát tối đa được chia. Nếu không truyền vào,
                                            sẽ dùng giá trị mặc định trong self.max_slice_nums.

        Returns:
            List[PIL.Image.Image]: Danh sách các ảnh đầu ra, gồm:
                - Ảnh `source_image` (ảnh gốc đã resize),
                - Các lát (patches) nếu ảnh được chia.
        """
        slice_images = []

        # Nếu không bật chế độ chia lát (slice_mode), chỉ trả về ảnh gốc 
        if not self.slice_mode:
            return [image]

        # Lấy số lát tối đa từ tham số hoặc thuộc tính mặc định 
        max_slice_nums = self.max_slice_nums if max_slice_nums is None else int(max_slice_nums)
        assert max_slice_nums > 0 

        # Gọi slice_image để resize/chia ảnh thành source + patches 
        source_image, patches, sliced_grid = self.slice_image(
            # default: 9  # default: 448  # default: 14
            image, max_slice_nums, self.scale_resolution, self.patch_size  
        )

        # Thêm ảnh chính (đã resize) vào danh sách
        slice_images.append(source_image)

        # Nếu có chia lát, thêm từng patch vào danh sách 
        if len(patches) > 0:
            for row in patches:
                for patch in row:
                    slice_images.append(patch)
        
        return slice_images 

    def get_sliced_grid(
        self,
        image_size: Tuple[int, int],
        max_slice_nums: int,
        nerver_split: bool = False
    ) -> Optional[Tuple[int, int]]:
        """
        Xác định lưới chia lát (grid) tốt nhất để cắt ảnh thành nhiều phần nhỏ, dựa trên tỷ lệ ảnh
        và số lượng lát tối đa cho phép.

        Args:
            image_size (Tuple[int, int]): Kích thước ảnh gốc (width, height).
            max_slice_nums (int): Số lượng lát (slices) tối đa cho phép.
            nerver_split (bool, optional): Nếu True, luôn không chia lát và trả về None.

        Returns:
            Optional[Tuple[int, int]]: Lưới chia lát tối ưu (số cột, số hàng),
                                    hoặc None nếu không cần chia lát.
        """
        original_width, original_height = image_size

        # Tính log tỉ lệ khung hình hình ảnh (để tìm lưới gần tương đương về tỉ lệ)
        log_ratio = math.log(original_width / original_height)

        # Calculate the ideal slice number (Tính toán độ "lớn" của ảnh so với độ lớn input của pretrained model)
        ratio = (original_width * original_height) / (self.scale_resolution * self.scale_resolution)

        # Số lượng lát cần chia là tỉ lệ diện tích, làm tròn lên, nhưng không vượt quá giới hạn
        multiple = min(math.ceil(ratio), max_slice_nums) 

        # Nếu ảnh quá nhỏ hoặc được yêu cầu không chia thì trả về None
        if multiple <= 1 or nerver_split:
            return None

        # Tìm danh sách các số lượng lát khả thi (trong khoảng ±1 so với multiple)
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # Tạo tất cả các cặp (cols, rows) có thể chia được từ số lát
        candidate_grids = []
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])  # cols x rows
                m += 1

        # Tìm lưới chia có tỉ lệ gần giống với tỉ lệ khung hình ban đầu nhất
        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))  # sai số log tỉ lệ
            if error < min_error:
                best_grid = grid
                min_error = error

        return tuple(best_grid)

    def get_slice_image_placeholder(
            self,
            image_size: Tuple[int, int],
            image_idx: int = 0,
            max_slice_nums: Optional[int] = None,
            use_image_id: Optional[bool] = None
        ) -> str: 
        """
        Tọa chuỗi placeholder (chuỗi văn bản mô phỏng ảnh) đại diện cho ảnh đã slice (chia lát), 
        phục vụ cho quá trình xử lý mô hình khi ảnh chưa thực sự được đưa vào (ví dụ khi chỉ 
        xử lý text hoặc training stub). 

        Args:
            image_size (Tuple[int, int]): Kích thước ảnh gốc (width, height).
            image_idx (int, optional): Chỉ số ảnh (dùng để chèn ID ảnh nếu cần). Mặc định là 0.
            max_slice_nums (Optional[int], optional): Số lát ảnh tối đa. Nếu không truyền, dùng giá trị mặc định.
            use_image_id (Optional[bool], optional): Có chèn ID ảnh vào chuỗi placeholder hay không.
                                                    Mặc định là self.use_image_id.

        Returns: 
            str: Chuỗi placeholder biểu diễn cho ảnh và lưới lát của ảnh nếu có.
                Bao gồm token bắt/đóng ảnh, ID (nếu cần), và các lát ảnh dạng text.
        """
        # Xác định số lát ảnh tối đa 
        max_slice_nums = self.max_slice_nums if max_slice_nums is None else int (max_slice_nums)
        assert max_slice_nums > 0 

        # Tìm lưới chia lát ảnh phù hợp cho ảnh 
        grid = self.get_sliced_grid(
            image_size=image_size,
            max_slice_nums=max_slice_nums
        )

        # Placeholder cơ bản cho một ảnh (dạng token): <image><unk>...</unk></image>
        image_placeholder = self.im_start_token + \
            self.unk_token * self.image_feature_size + \
            self.im_end_token

        # Nếu yêu cầu chèn ID ảnh, thêm token <image_id>...</image_id>
        use_image_id = self.use_image_id if use_image_id is None else bool(use_image_id)
        if use_image_id:
            final_placeholder = self.get_image_id_placeholder(image_idx) + image_placeholder
        else:
            final_placeholder = image_placeholder 
        
        # Nếu đang bật chế độ chia lát (slice mode), thêm chuỗi placeholder cho grid lát 
        if self.slice_mode:
            final_placeholder += self.get_grid_placeholder(grid=grid)
        
        return final_placeholder 

    def to_pil_image(
            self, 
            image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], 
            rescale: bool=None
        ) -> PIL.Image.Image:
        """
        Converts `image` to a PIL Image. 
        Optionally rescales it and puts the channel dimension back as the last axis if needed.

        Args:
            image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`):
                The image to convert to the PIL Image format.
            rescale (`bool`, *optional*):
                Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will
                default to `True` if the image type is a floating type, `False` otherwise.
        
        Returns:
            PIL.Image.Image
        """
        if isinstance(image, PIL.Image.Image):
            return image
        if is_torch_tensor(image):
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if rescale is None:
                # rescale default to the array being of floating type.
                rescale = isinstance(image.flat[0], np.floating)
            # If the channel as been moved to first dim, we put it back at the end.
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            if rescale:
                image = image * 255
            image = image.astype(np.uint8)
            return PIL.Image.fromarray(image)
        return image

    def reshape_by_patch(self, image: np.ndarray) -> np.ndarray:
        """
        Biến ảnh thành dạng [C, patch_size, num_patches], nghĩa là chia 
        theo từng dòng patch — thường dùng khi MiniCPM-o nạp từng patch như một 
        chuỗi token liên tục.

        Args:
            image (np.ndarray): Ảnh đầu vào dạng Numpy array với shape [3, H, W], 
                                tức là có 3 kênh (RGB) và chiều cao/rộng tùy ý. 

        Returns:
            np.ndarray: [3, patch_size, num_patches],
                trong đó num_patches = (H * W) // (patch_size^2)
        """
        # Chuyển ảnh từ Numpy sang Tensor 
        image_tensor = torch.from_numpy(image)
        patch_size = self.patch_size 

        # Dùng unfold để cắt ảnh thành các patch (flatten theo từng vòng)
        # Kết quả shape: [C, patch_size * patch_size, num_patches]
        patches = torch.nn.functional.unfold(
            image_tensor, 
            (patch_size, patch_size), 
            stride=(patch_size, patch_size)
        )

        # Chuyển shape về [C, patch_size, patch_size, num_patches]
        patches = patches.reshape(image_tensor.size(0), patch_size, patch_size, -1) 

        # Đưa patch thành [C, patch_size, num_patches] bằng cách gộp lại 2 chiều cuối
        patches = patches.permute(0, 1, 3, 2).reshape(image_tensor.size(0), patch_size, -1)

        # Chuyển lại về Numpy Array để tương thích với các bước xử lý sau 
        return patches.numpy()

    def preprocess(
            self, 
            images: Union[PIL.Image.Image, List[PIL.Image.Image], List[List[PIL.Image.Image]]], 
            do_pad: Optional[bool] = True, 
            max_slice_nums: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            **kwargs, 
        ) -> KhaanhBatchFeature:
        """
        Tiền xử lý ảnh đầu vào để chuẩn bị cho mô hình:
            - chuyển ảnh về định dạng chuẩn
            - chia lát nếu cần 
            - chuẩn hóa và chia patch 
            - trả về batch dạng đặc biệt chứa ảnh đã xử lý 

        Args:
            images (Union[PIL.Image.Image, List[PIL.Image.Image], List[List[PIL.Image.Image]]]):
                Ảnh đầu vào (1 ảnh, 1 danh sách ảnh, hoặc danh sách các danh sách ảnh).
            do_pad (bool, optional): Không sử dụng trong hàm này, để tương thích với chuẩn `ImageProcessor`.
            max_slice_nums (int, optional): Số lát tối đa mỗi ảnh có thể được chia.
            return_tensors (str or TensorType, optional): Loại tensor mong muốn cho output ('pt', 'np', etc).

        Returns:
            KhaanhBatchFeature: Batch đặc biệt chứa các tensor ảnh, kích thước ảnh gốc, và kích thước patch mục tiêu.
        """
        # Chuẩn hóa input thành danh sách 2 chiều: List[List[PIL.Image.Image]]
        if isinstance(images, PIL.Image.Image):
            images_list = [[images]]
        elif isinstance(images[0], PIL.Image.Image):
            images_list = [images]
        else:
            images_list = images

        new_images_list = []
        image_sizes_list = []
        tgt_sizes_list = []

        for _images in images_list:
            if _images is None or len(_images) == 0:
                new_images_list.append([])
                image_sizes_list.append([])
                tgt_sizes_list.append([])
                continue
        
            if not valid_images(_images):
                raise ValueError(
                    "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                    "torch.Tensor, tf.Tensor or jax.ndarray."
                )

            # Đảm bảo ảnh có định dạng PIL và RGB
            _images = [self.to_pil_image(image).convert("RGB") for image in _images]

            # Xác định thứ tự kênh (CHW hay HWC) từ ảnh đầu tiên
            input_data_format = infer_channel_dimension_format(np.array(_images[0]))

            new_images = []
            image_sizes = [image.size for image in _images]
            tgt_sizes = []

            for image in _images:
                # Lấy danh sách các lát ảnh (gồm ảnh gốc đã resize + patches)
                image_patches = self.get_sliced_images(image, max_slice_nums)

                # Chuyển sang NumPy, scale pixel về [0,1]
                image_patches = [to_numpy_array(image).astype(np.float32) / 255 for image in image_patches]
                
                # Chuẩn hóa giá trị pixel theo mean/std
                image_patches = [
                    self.normalize(
                        image=image, mean=self.mean, 
                        std=self.std, input_data_format=input_data_format
                    )
                    for image in image_patches
                ]
                # Đảm bảo thứ tự kênh là Channel-first: [C, H, W]
                image_patches = [
                    to_channel_dimension_format(
                        image, ChannelDimension.FIRST, 
                        input_channel_dim=input_data_format
                    )
                    for image in image_patches
                ]

                # Chia từng lát ảnh thành patch, lưu lại 
                for slice_image in image_patches:
                    new_images.append(self.reshape_by_patch(slice_image))
                    tgt_sizes.append(
                        np.array(
                            (slice_image.shape[1] // self.patch_size, slice_image.shape[2] // self.patch_size)
                        )
                    )

            if tgt_sizes:
                tgt_sizes = np.vstack(tgt_sizes) # Stack các kích thước patch thành 2D array 

            new_images_list.append(new_images)
            image_sizes_list.append(image_sizes)
            tgt_sizes_list.append(tgt_sizes)


        # Trả về batch dạng KhaanhBatchFeature chứa pixel_values, image_sizes, tgt_sizes
        return KhaanhBatchFeature(
            data={
                "pixel_values": new_images_list,
                "image_sizes": image_sizes_list,
                "tgt_sizes": tgt_sizes_list
            },
            tensor_type=return_tensors,
        )

AutoImageProcessor.register("KhaanhImageProcessor", KhaanhImageProcessor)
