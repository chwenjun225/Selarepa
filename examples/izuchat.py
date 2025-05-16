# Cấu trúc dữ liệu là cách tổ chức và lưu trữ dữ liệu trong bộ nhớ sao cho truy cập và xử lý hiệu quả nhất 
# Tuyến tính - List, Stack, Queue, Deque 
# Phi tuyến tính - Tree, Graph, Tree - Dùng cho hệ thống phân cấp tìm kiếm  
# Hash-based - Dict, Set, HashTable - Truy cập cực nhanh 
# Tuần tự hóa dữ liệu - LinkedList - Dễ chèn/xóa, nhưng khó truy cập nhanh 


# 🔧 1 List (Danh sách )
# a = [1, 2, 3]
# a.append(4)   # [1,2,3,4]
# a[1]          # -> 2
# Truy cập nhanh: O(1)
# Chèn/xóa đầu chậm: O(n)


# 🔧 2. Stack (Ngăn xếp – LIFO)
# stack = []
# stack.append(1)
# stack.append(2)
# stack.pop()  # => 2
# Sử dụng khi xử lý theo kiểu: vào sau ra trước

# Dùng trong đệ quy, duyệt cây, undo/redo...


# 🔧 3. Queue & Deque
# from collections import deque
# q = deque()
# q.append(1)       # thêm vào cuối
# q.popleft()       # xoá đầu
# Queue: FIFO – vào trước ra trước

# Deque: 2 đầu → nhanh cả appendleft, popright


# 🔧 4. Dict (Từ điển – Hash Table)
# d = {"agent": "alpha", "id": 1}
# d["agent"]        # => "alpha"
# Truy cập siêu nhanh O(1)

# Dùng để ánh xạ, cấu hình, lưu trạng thái agent...


# 🔧 5. Set (Tập hợp – không trùng lặp)
# s = {1, 2, 3}
# s.add(2)          # không thêm vì đã có
# Rất nhanh khi kiểm tra phần tử có tồn tại không