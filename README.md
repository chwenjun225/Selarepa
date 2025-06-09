**System and Method for Self-Labeling Repair Using LLM-based Agent in Visual Violation Detection**

### 🎯 Mục tiêu & Điểm khác biệt bằng sáng chế:

(1) Tùy chỉnh xây dựng một mô hình Multi-Modal Large Language Model (MLLM) được huấn luyện chuyên biệt cho nhiệm vụ theo dõi hành vi vi phạm của công nhân trong nhà máy;
(2) Mô hình có kích thước gọn nhẹ, tốc độ xử lý nhanh và hoạt động hiệu quả trên GPU RTX 4090 nhờ được thiết kế từ đầu thay vì chỉ tinh chỉnh từ mô hình có sẵn.

### 📎 Tóm tắt:

Phát minh này mô tả một hệ thống sử dụng AI agent dựa trên mô hình ngôn ngữ lớn (LLM) để tự động sửa các nhãn dữ liệu sai được tạo ra bởi hệ thống camera giám sát công nghiệp, nhằm giảm thiểu sai số đầu vào cho mô hình học sâu trong bài toán phát hiện hành vi vi phạm.

### 🏗 Cấu trúc chi tiết:

* Field of invention: AI, computer vision, label correction.
* Background: camera systems often have false positives in behavior classification.
* Technical solution:

* Multi-stage pipeline: object detection → patch extraction → label analysis by LLM-agent → feedback loop for retraining.
* Advantages: giảm nhân lực, tăng tốc độ làm sạch dữ liệu, cải thiện độ chính xác.


```
"""
for line in label_lines: print(line)
    0 0.8542 3701.68 1267.73 3785.19 1389.47
    0 0.8300 13.33 1260.46 99.49 1391.39
    0 0.8158 867.47 1418.24 1138.22 1621.52
    0 0.8137 2622.15 1521.62 3176.85 1792.72
    3 0.8077 990.19 1434.79 1103.89 1529.90
    0 0.8067 3581.71 1233.21 3662.01 1393.82
    3 0.7976 44.94 1266.51 79.93 1338.17
    3 0.7964 152.05 1269.31 189.75 1346.87
    0 0.7889 755.08 1596.39 1261.03 1805.51
    3 0.7661 3731.03 1270.31 3768.96 1331.84
    3 0.7478 2951.10 1552.99 3099.83 1646.09
    0 0.7332 851.98 1117.93 941.66 1331.20
    0 0.7204 2747.91 1355.84 2973.67 1558.98
    0 0.7131 3470.71 1229.90 3556.78 1449.99
    0 0.7047 135.16 1264.29 201.13 1383.53
    3 0.6989 3499.60 1232.66 3532.69 1290.13
    2 0.6906 911.35 1296.02 952.38 1328.29
    3 0.6547 3595.15 1238.36 3632.77 1293.65
    0 0.6416 2734.71 1173.49 2824.40 1319.42
    0 0.6401 2811.00 1245.41 2958.84 1367.18
    0 0.6388 3019.23 1164.07 3090.83 1326.48
    0 0.6378 508.39 1059.88 533.86 1204.03
    0 0.6344 1997.56 1685.41 2677.14 1917.66
    0 0.6125 563.89 1094.34 599.94 1268.24
    0 0.5955 810.46 1065.07 850.98 1267.52
    0 0.5821 2822.14 1258.79 2927.14 1368.19
    0 0.5751 2689.42 1144.10 2729.64 1250.19
    3 0.5721 3404.46 1209.56 3440.51 1274.31
    0 0.5710 991.02 1067.02 1040.15 1238.71
    0 0.5669 2007.47 1679.35 3137.38 1915.96
    3 0.5635 2699.27 1144.74 2721.76 1196.50
    3 0.5174 816.24 1067.13 839.50 1116.64
    3 0.4615 902.23 1621.37 1068.44 1705.13
    3 0.4396 2771.86 1174.48 2792.02 1223.92
    0 0.4143 2820.50 1219.94 2972.27 1356.76
"""
```