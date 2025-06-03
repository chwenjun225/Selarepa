Hai dòng này là **import các cấu trúc dữ liệu mạnh mẽ có sẵn trong Python**, cụ thể là `deque` và `heapq`, rất hữu ích cho thuật toán và xử lý dữ liệu hiệu quả:

---

### ✅ `from collections import deque`

* **`deque`** (double-ended queue): là **hàng đợi hai đầu**, cho phép **thêm/xóa phần tử ở cả đầu và cuối** hiệu quả hơn `list`.

#### 📌 Tại sao dùng `deque` thay vì `list`?

* `list.pop(0)` hoặc `list.insert(0, x)` tốn **O(n)** (chậm)
* `deque.popleft()` và `deque.appendleft()` tốn **O(1)** (rất nhanh)

#### 🔧 Ví dụ:

```python
from collections import deque

dq = deque()
dq.append(1)         # thêm vào cuối
dq.appendleft(2)     # thêm vào đầu
print(dq)            # deque([2, 1])
dq.pop()             # xóa cuối → 1
dq.popleft()         # xóa đầu → 2
```

👉 **Ứng dụng**: BFS, sliding window, undo/redo, task queue,…

---

### ✅ `import heapq`

* **`heapq`**: module dùng để làm việc với **heap (priority queue)** – mặc định là **min-heap** (phần tử nhỏ nhất ở đầu)

#### 🔧 Ví dụ:

```python
import heapq

nums = [5, 3, 8, 1]
heapq.heapify(nums)     # biến list thành heap: [1, 3, 8, 5]
heapq.heappush(nums, 0) # thêm phần tử → heap vẫn được sắp xếp
x = heapq.heappop(nums) # lấy ra phần tử nhỏ nhất (0)
```

👉 **Ứng dụng**: Dijkstra, A\*, tìm k phần tử nhỏ nhất/lớn nhất,…

---

### ✅ So sánh nhanh:

| Module  | Dùng cho            | Đặc điểm nổi bật                        |
| ------- | ------------------- | --------------------------------------- |
| `deque` | Queue/Stack 2 chiều | Thêm/xóa cực nhanh ở 2 đầu              |
| `heapq` | Priority Queue      | Luôn truy cập phần tử nhỏ nhất đầu tiên |

---

Nếu bạn muốn, mình có thể viết ví dụ cụ thể áp dụng cho một agent hoặc thuật toán sử dụng 2 cấu trúc này. Bạn đang dùng chúng vào phần nào?
