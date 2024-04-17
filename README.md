# Quality-Assurance
### Extra library installation required
```
pip install openpyxl
pip install pypdf2
pip install reportlab
```

## Running Example

```python
python evaluate.py --run_dir <project_name>
```

(1) mode : only "classification" supported

(2) data : "choi" / "snu" / "fit" / "plco" / "stpeter" / "rsna"

- choi : indovnn dataset annotated by Dr.Choi that contains 4085 examples --> internal dataset
- snu : aws dataset annotated by snu drs that contains 1980 examples --> internal dataset
- fit : TB dataset that contains 1032 examples --> external dataset
- plco : lung cancer datset that contains 254 examples --> external dataset
- stpeter : TB datset that contains 337 examples --> external dataset
- rsna : pnuemonia datsaet that contains 625 examples --> external dataset

(3) out_json_path



- AI output json path for each datasets
- your json should look like this (containing file_name and the output scores)

![image](https://github.com/RadiSen-AI/Quality-Assurance/assets/81098548/e5148687-7dd0-44dc-830f-af21fb7372ed)
  
- The outputs' keys are the pre-defined class index

```
categories = [
            {'id': 1, 'name': 'atelectasis', 'supercategory': 'atelectasis'},
            {'id': 2, 'name': 'lung opacity', 'supercategory': 'lung opacity'},
            {'id': 3, 'name': 'effusion', 'supercategory': 'effusion'},
            {'id': 4, 'name': 'nodule mass', 'supercategory': 'nodule mass'},
            {'id': 5, 'name': 'hilar', 'supercategory': 'hilar'},
            {'id': 6, 'name': 'fibrosis', 'supercategory': 'fibrosis'},
            {'id': 7, 'name': 'pneumothorax', 'supercategory': 'pneumothorax'},
            {'id': 8, 'name': 'cardiomegaly', 'supercategory': 'cardiomegaly'},
            {'id': 9, 'name': 'edema', 'supercategory': 'lung opacity'},
            {'id': 10, 'name': 'nodulemasswocavitation', 'supercategory': 'nodule mass'},
            {'id': 11, 'name': 'cavitarynodule', 'supercategory': 'nodule mass'},
            {'id': 12, 'name': 'miliarynodule', 'supercategory': 'nodule mass'},
            {'id': 13, 'name': 'fibrosisinfectionsequelae', 'supercategory': 'fibrosis'},
            {'id': 14, 'name': 'fibrosisild', 'supercategory': 'fibrosis'},
            {'id': 15, 'name': 'bronchiectasis', 'supercategory': 'bronchiectasis'},
            {'id': 16, 'name': 'emphysema', 'supercategory': 'emphysema'},
            {'id': 17, 'name': 'subcutaneousemphysema', 'supercategory': 'emphysema'},
            {'id': 18, 'name': 'pleuralthickening', 'supercategory': 'pleuralthickening'},
            {'id': 19, 'name': 'pleuralcalcification', 'supercategory': 'pleuralcalcification'},
            {'id': 20, 'name': 'medical device', 'supercategory': 'medical device'},
            {'id': 101, 'name': 'normal', 'supercategory': 'normal'},
            {'id': 102, 'name': 'pneumonia', 'supercategory': 'pneumonia'},
            {'id': 103, 'name': 'tuberculosis', 'supercategory': 'tuberculosis'},
            {'id': 104, 'name': 'others', 'supercategory': 'others'},
            {'id': -1, 'name': 'discard', 'supercategory': 'discard'}
        ]
```



(4) run_dir

- name of project
- result file is saved at "/projects"


## Notice

- Current version of QA process outputs 10-class classficiation preformance evaluation only.
- PDF file is automatically generated at your project directory

## TO DO

- Detection performance evaluation
- Finding and Disease separation
