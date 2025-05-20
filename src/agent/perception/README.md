đã xong module cognition, giờ sẽ sang perception 

ở module perception này ta sẽ thiết kế một module nhằm đưa ra các observation đầy đủ về thế giới quan mà agent nhìn hoặc cảm nhận được.

vậy các module này là gì 

(1) Perception: nếu nhìn được, nhận ảnh đầu vào -> đưa vào llama-3.2-11b-vision-instruct để sinh ra mô tả về bức ảnh nó nhìn thấy, kết quả của bước này được gọi là observation; 
    bất kể thông tin dù là text hoặc image thì đều phải đi qua một bộ sinh cảm nhận observation. Sau đó mới lấy observation đưa vào module sau 

(2) lấy observation đưa vào cognition module 