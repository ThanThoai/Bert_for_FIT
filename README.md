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
Tải dataset, pretrain BERT và để vào thư mục cần thiết. Link pretrain model và link dataset. Cấu trúc thư mục:
 ```bash
  |--dataset
  |
  |--model
  |  |
  |  |--data
  |  |   |-train-{model_name}.pt
  |  |   |-valied-{model_name}.pt
  |  |   
  |  |-src
  |
  |--pretrained
  |  |
  |  |--bert-base-uncased:
  |  |       |
  |  |       |--pytorch_model.bin
  |  |       |--config.json
  |  |       |--vocab.txt
  |  |    
  |  |--bert-large-uncased:
  |  |       |
  |  |       |--pytorch_model.bin
  |  |       |--config.json
  |  |       |--vocab.txt

```  

Chỉnh sửa file config.yaml để điều chỉnh các tham số. Sau đó chạy lệnh: 
```bash
python main.py
```

## How to test

Tải các model đã huấn luyện. Chỉnh sửa đường dẫn model trong file test.py
```bash
python test.py
```

  
