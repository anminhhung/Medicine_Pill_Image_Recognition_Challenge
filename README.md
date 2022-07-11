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
  │
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
# python3 demo.py --config-file configs/setup.yaml --input ./data/pills --models models/model_12345.pth

python demo.py
```






