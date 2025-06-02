**System and Method for Self-Labeling Repair Using LLM-based Agent in Visual Violation Detection**

### 📎 Tóm tắt:

Phát minh này mô tả một hệ thống sử dụng AI agent dựa trên mô hình ngôn ngữ lớn (LLM) để tự động sửa các nhãn dữ liệu sai được tạo ra bởi hệ thống camera giám sát công nghiệp, nhằm giảm thiểu sai số đầu vào cho mô hình học sâu trong bài toán phát hiện hành vi vi phạm.

### 🏗 Cấu trúc chi tiết:

* Field of invention: AI, computer vision, label correction.
* Background: camera systems often have false positives in behavior classification.
* Technical solution:

* Multi-stage pipeline: object detection → patch extraction → label analysis by LLM-agent → feedback loop for retraining.
* Advantages: giảm nhân lực, tăng tốc độ làm sạch dữ liệu, cải thiện độ chính xác.
