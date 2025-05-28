# 1. Visual Encoder

- Nhiệm vụ: Chuyển đổi ảnh RGB thô thành một chuỗi "Visual tokens" có thể xử lý bởi phần ngôn ngữ

- Cách làm: Sử dụng SigLIP SoViT-400m/14 - một biến thể Vision Transformers nhẹ đã được huấn luyện trước để trích xuất đặc trưng hình ảnh.

- Kỹ thuật: "Adaptive visual encoding" nghĩa là encoder có thể tự động điều chỉnh tỷ lệ down-sampling hoặc cách cắt patch để giữ lại thông tin cần thiết, đặc biệt với ảnh độ phân giải cao. 


# 2. Compression Layer 

- Nhiệm vụ: Giảm số lượng visual tokens xuống mức vừa phải để LLM không phải xử lý một lượng quá lớn dữ liệu (vừa tiết kiệm tính toán, vừa giữ được thông tin quan trọng).

- Cách làm: Dùng perceiver Resampler - một cấu trúc mà ở đây chỉ dùng một layer cross-attention:

(1) Cross-attention: lấy một tập query (có kích thước cố định, ví dụ 64 token) attend lên toàn bộ visual tokens (key/value) để "tóm gọn" chúng. 

(2) Kết quả là một tập compressed tokens có kích thước nhỏ hơn, nhưng vẫn hội tụ đủ thông tin đại diện cho cả ảnh. 


# 3. Large Language Model 

- Nhiệm vụ: Sinh văn bản điều kiện dựa trên cả compressed visual tokens và đầu vào văn bản (prompt)

- Cách làm: Ghép chuỗi token của ảnh (sau compression) vào trước hoặc sau token của prompt text, rồi chạy quá kiến trúc Transformer của LLM để sinh tiếp phần text trả lời. 


# 👉 Tóm lại luồng dữ liệu

Input Image  ──► Visual Encoder (SigLIP SoViT) ──► visual tokens
                                 │
                                 ▼
                    Compression Layer (Perceiver Resampler)
                                 │
                                 ▼
           [compressed visual tokens] + [prompt text tokens]
                                 │
                                 ▼
                     Large Language Model ──► Generated Text


- Mỗi thành phần làm một việc chuyên biệt:

1. Nhìn (Visual Encoder)

2. Tóm gọn (Compression Layer)

3. Nói (LLM)

> Kiến trúc này cho phép MiniCPM-V vừa mạnh về thị giác (nhờ SoViT), vừa gọn nhẹ để chạy trên thiết bị biên (nhờ compression chỉ một layer), đồng thời tận dụng sức mạnh sinh văn bản tiên tiến của LLM gốc (Llama3).



















