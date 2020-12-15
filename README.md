<div align="center">    
 
# BERT for FIT - Filling In The blank     
 
</div>
 
## Description   
Xây dựng mô hình dự đoán từ cần điền vào chỗ trống và áp dụng vào giải đề thi Toiec.  

## How to run   
Đầu tiên, cài đặt những thứ cần thiết:   
```bash
# clone project   
git clone https://github.com/ThanThoai/Bert_for_FIT

# install project   
cd Bert_for_FIT
pip install -r requirements.txt
 ```   
Tải dataset, pretrain BERT và để vào thư mục cần thiết. 
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```
Chỉnh sửa file config.yaml để điều chỉnh các tham số. Sau đó chạy lệnh: 
```bash
python main.py
```

  
