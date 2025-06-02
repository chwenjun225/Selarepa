# Workflow Ứng dụng hệ thống LLM-Agent sửa nhãn trong giám sát hành vi công nhân tại dây chuyền sản xuất Foxconn 

**1 Giai đoạn thu thập hình ảnh (Monitoring Layer)**
- Camera 360 được lắp đạt dọc theo dây chuyền sản xuất, liên tục ghi lại hình ảnh của công nhân tại các trạm thao tác
- Video được chia thành các frame theo thời gian thực hoặc theo chu kỳ (mỗi 0.5s-1s/frame) tùy thuộc vào tốc độ dây chuyền.

**2 Giai đoạn phát hiện hành vi ban đầu (Detection Layer)**
- Mỗi frame được đưa qua mô hình YOLO (hoặc mô hình tương đương) để phát hiện hành vi vi phạm như
(a) ngồi sai tư thế, 
(b) không đội mũ bảo hộ, 
(c) cúi người quá mức, 
(d) ra khỏi vùng thao tác cho phép, 
- Mỗi vùng phát hiện được phát hiện bằng bounding-box, với nhãn tạm thời (`label` và `confidence score`).

**3 Giai đoạn xử lý patch ảnh (Pre-processing layer)**
- Từ bounding-box, hệ thống cắt ra patch ảnh tương ứng. 
- Mỗi patch kèm theo (a) `label` tạm thời do YOLO dự đoán, (b) `timestamp`, `camera_id`, `position_id`, `confidence`.

**4 Giai đoạn phân tích bởi LLM-Agent (Correction Layer)**
- Các patch này được truyền vào LLM-based Agent, mô hình ngôn ngữ có khả năng phân tích cả hình ảnh lẫn ngữ cảnh. 
- Agent sẽ:
(1) kiểm tra tính chính xác của label hiện tại
(2) nếu đúng -> giữ nguyên 
(3) nếu sai -> đề xuất nhãn đúng + giải thích ngắn gọn (ví dụ: "cúi người do nhặt vật, không vi phạm"). 

**5 Giai đoạn ghi lại và huấn luyện lại (Feedback Layer)**
- Kết quả được đưa vào một dataset hiệu chỉnh, gồm (1) ảnh patch, (2) Label cũ và label mới (nếu sửa), (3) Log reasoning từ agent (giải thích ngắn).
- Dataset này được dùng để (1) huấn luyện lại mô hình YOLO (fine-tune), Hoặc làm tập benchmark kiểm định.
