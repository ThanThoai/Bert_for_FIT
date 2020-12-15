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
Tải dataset, pretrain BERT và để vào thư mục cần thiết. Link [BERT_pretrain](https://drive.google.com/drive/folders/171GhawHqUuOhnSFNl9AtQF1WPRqI-Ehf?usp=sharing) và [dataset](https://drive.google.com/drive/folders/1C1GsV1MiDWa8NRFy-pHX8d5V-lKtIo_1?usp=sharing). Cấu trúc thư mục:
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

Mô hình được huấn luyện trên GPU Tesla V100 16GB với batch_size = 2.

## How to test

Tải các [model](https://drive.google.com/drive/folders/1pCOcSAe0BVPey_UKMRTaT5YQ9sbj9R3W?usp=sharing) đã huấn luyện. Chỉnh sửa đường dẫn model trong file test.py
```bash
python test.py
```

  
