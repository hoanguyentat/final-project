# Final Project
Ứng dụng mạng DenseNet trong bài toán phân loại giới tính
##Chuẩn bị dữ liệu
Dữ liệu được lấy về từ bộ dataset của ``IMDB-WIKI – 500k+ face images with age and gender labels`` bao gồm 68138 ảnh khuôn mặt người.
Dataset tải về được lưu trong thư mục data (nằm trong thư mục gốc)
## Require
```python3```
```tensorflow```
```opencv```

##Run
### Parametters
Các tham số trong quá trình training có thể thay đổi trong file parametters.py

###Generate tfrecords file
```python3 convert_to_tfrecored.py  --db wiki --img_size 64```

Lưu các file được tạo ra trong thư mục tfrecords nằm trong thự mục gốc

### Training
```python densenet _ gender_ with_queue.py```

Sau khi chạy xong chương trình sinh ra file ``checkpoint`` trong thư mục ```model-gender-new``` 

### Prediction

```python predict.py``` (tham số đầu vào là đường dẫn của thư mục dự  cần dự đoán và thư mục chứa checkpoint) 