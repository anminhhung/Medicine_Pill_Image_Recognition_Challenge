<div align="center">
<img src="https://vcdn-vnexpress.vnecdn.net/2022/07/05/nhan-dang-thuoc-3644-165585973-3102-9866-1657005587.jpg" width="30%">
<img src="https://i.ytimg.com/vi/p-Nn0RgwudE/mqdefault.jpg" width="35%">

</div>
<h1>AIO Vision - Medicine Pill Recognition</h1>

- ***VAIPE***  : Medicine Pill Image Recognition Challenge - AI4VN 2022

- ***Meeting***: Every Monday & Thursday night time (8h30 PM)


## Log Team meeting:
+ 2 Challenges team for competition
+ 4 Premilary tasks: 
```
     |---- 1. Object detection/segmemtaion 
     |---- 2. OCR
     |---- 3. Data Augmentation
     |---- 4. Classifcation
```


+ Folder Structure

  ```
  Main-folder/
  │
  ├── config/ 
  │   ├── config.py - configuration
  │
  ├── data/ - default directory for storing input data
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  |
  ├── nets/ - this folder contains any net of your project.
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging 
  │   └── submission/ -  submission file are saved here
  │
  ├── scripts/ - main function 
  │   └── pipeline.py
  │   └── OCR.py
  │   └── segment.py
  |
  ├── test/ - test functions
  │   └── run.py
  │   └── ...
  |
  ├── tools/ - open source are saved here
  │   └── detectron2 dir
  │   └── ...
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```
## Pretrained models will be stored in Google Drive!

Link pretrained models: [Link GG drive](https://drive.google.com/drive/u/0/folders/1IQZZ5XPQfUKYhjZxisoazkNoHUT6qEap)


## 🥰 Demo
Run a quick demo would be like:

```python 
#python3 demo.py --config-file configs/setup.yaml --input ./data/pills --models models/model_12345.pth
python demo.py
```


## 🔥 Dataset

Original 

Website: https://vaipe.org/#challenge
Train: [Link GG drive](https://drive.google.com/drive/folders/1F7JvhcAIzZews4u8Cba_HntUZk25jQdh)
Test : [Link GG drive](https://drive.google.com/file/d/146BJ1ER43mOUS7IL4Ewgs2vaAylCXt2l/view?fbclid=IwAR2kZtM6YrtvaiZisWZdBB69_mBYRs2BI_jWDLvtaMZ-6j-vAq6da5jpP0E)
+ File train: có 9502 ảnh thuốc(kèm json) + 1173 ảnh đơn thuốc.
+ File test: có 1173 ảnh thuốc(kèm json) + 172 ảnh đơn thuốc)


Kaggle Re-up
```
os.environ['KAGGLE_USERNAME'] = “tên user của em” # username from the json file
os.environ['KAGGLE_KEY'] = “key tài khoản" # key from the json file
!kaggle datasets download -d tommyngx/vaipepill2022
```

## Merge 108 classes -> 88 classes

```
# Load hash table
# Link: ./Medicine_Pill_Image_Recognition_Challenge/Data/pill_categories/label88.npy
read_dictionary = np.load('label88.npy',allow_pickle='TRUE').item()
print(read_dictionary['1']) # nhập class cũ để xuất class mới!
```


