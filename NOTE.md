![KhaAnh](./assets/logo-transparent.png)

# 1 Đặt vấn đề
Để tiến gần hơn tới mục tiêu Trí tuệ nhân tạo tổng quát (AGI), một trong những hướng nghiên cứu tiềm năng là trang bị cho mô hình ngôn ngữ lớn (LLM) các năng lực nhận thức cao cấp như khả năng tư duy giải quyết vấn đề (ví dụ: Chain-of-Thought, Reflection, Reasoning, ReAct, v.v.) và khả năng truy vấn – làm giàu ngữ cảnh để tối ưu hóa phản hồi (Retrieval-Augmented Generation). Mặc dù đã có những tiến bộ đáng kể, các ứng dụng thực tế của LLM hiện vẫn còn hạn chế, chủ yếu dừng lại ở các tác vụ hỏi đáp trong lĩnh vực như chatbot, y tế, hay pháp lý, do độ chính xác chưa cao và mô hình vẫn chỉ đơn thuần dự đoán token kế tiếp mà chưa thật sự "hiểu" ngữ cảnh. Để mở rộng khả năng ứng dụng sang các lĩnh vực đòi hỏi nhận thức thời gian và ngữ cảnh như AI-native trong mạng 5G/6G hoặc an ninh mạng, LLM cần có khả năng hiểu được các hiện tượng theo thời điểm (chẳng hạn: trong khoảng 19h–21h tại Việt Nam là giờ cao điểm dẫn đến nghẽn mạng) và phân biệt được nguyên nhân (người dùng truy cập tăng hay bị tấn công mạng). Những yêu cầu này đòi hỏi một hệ thống AI có năng lực nhận thức bối cảnh và đưa ra hành động phù hợp. Bằng sáng chế này, tiết lộ một hệ thống trí tuệ nhân tạo bao gồm phần cứng máy tính và thuật toán phần mềm có khả năng xử lý các bài toán trong lĩnh vực an ninh mạng và viễn thông thế hệ mới (6G), được xây dựng theo kiến trúc multi-agent lấy cảm hứng từ khoa học thần kinh, nhằm tự động hóa các tác vụ phức tạp trong các hệ thống này.


# 2 Describe the past solutions and disadvantages 

## 2.1 Hướng nhìn tổng quan về các tác phẩm nghiên cứu liên quan

**Hiện nay, các công trình nghiên cứu về trí tuệ nhân tạo lấy cảm hứng từ não bộ (brain-inspired AI) chủ yếu tập trung vào các hướng như sau**

1. Deep Learning + Neuromorphic: Tập trung vào việc thiết kế phần cứng và hệ thống tính toán mô phỏng cấu trúc và cơ chế hoạt động của não người — bao gồm neuron, synapse, và cơ chế xử lý thông tin dựa trên xung điện (spike-based) và học thích nghi. Hướng này thường sử dụng chip chuyên dụng thay cho GPU truyền thống. Tuy nhiên, cơ chế phần mềm phương pháp này vẫn là các phương pháp Deep Learning. 
2. Deep Learning + Neuroscience: Xây dựng các module phần mềm lấy cảm hứng từ cấu trúc não bộ, kết hợp giữa kiến thức khoa học thần kinh và các kỹ thuật deep learning, triển khai trên GPU để học và suy luận.
3. Multi-Agent + Neuroscience: Phát triển các hệ thống phần mềm gồm nhiều tác tử nhận thức (cognitive agents), mỗi tác tử mô phỏng một chức năng của não bộ. Các tác tử này phối hợp với nhau dựa trên nguyên lý hoạt động thần kinh, vận hành theo kiến trúc multi-agent systems và được triển khai trên GPU.

**Hiện nay, các công trình nghiên cứu về mạng 6G chủ yếu tập trung vào các hướng như sau**
- Cần tìm đọc các bài báo liên quan đến mạng 6G khoảng 30 bài
1. AI-Native: Xây dựng các hệ thống trí tuệ nhân tạo tích hợp vào trong mạng 6G, tự động hóa hoàn toàn các thao tác mạng 6G
2. ...
3. ...

**Hiện này, các công trình nghiên cứu về an ninh mạng chủ yếu tập trung vào các hướng sau**
- Cần tìm đọc các bài báo liên quan đến anh ninh mạng khoảng 30 bài
1. ...
2. ...
3. ...

**Hiện nay, các bằng sáng chế liên quan chủ yếu tập trung vào các hướng sau**
- Cần tìm đọc các bằng sáng chế liên quan khoảng 30 bài
1. ...
2. ...
3. ...

