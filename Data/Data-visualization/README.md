# Visualize Data

Data link: https://www.kaggle.com/datasets/tommyngx/vaipepill2022
## Structuring files
To get it run, you need the file structure as below
```
  ./
  │
  ├── font/ 
  │   ├── SVN-Arial-Regular.ttf - Vietnamese-supported font
  │
  ├── public_test/ 
  │   ├── pill/
  │   ├── prescription/
  │   ├── pill_pres_map.json
  │
  ├── public_train/ 
  │   ├── pill/
  │   ├── prescription/
  │
  ├── visualize_data.py - file contains util functions to visualize labeled prescription and pill images   
  │
  ├── main.py - run here

```

## Usage
In main.py, visualize the prescription "VAIPE_P_TRAIN_1127"

```python
from visualizeData import visualize_prescription
visualize_prescription("VAIPE_P_TRAIN_1127.json", visualize_pills=True)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)