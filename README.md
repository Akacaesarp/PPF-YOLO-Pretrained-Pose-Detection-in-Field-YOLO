# PPF-YOLO:Pretrained Pose Detection in Field YOLO

PPF-YOLO is a computationally lightweight pose detection model for drone-captured field maize imagery. 

## Dataset
The dataset used in this study can be accessed at: http://phenomics.agis.org.cn/#/category.
In the MIPDB database, annotations and images are stored in JSON format. You need to use data_preprocess/data_preprocess.py to convert the JSON files to the YOLO format:
## Dataset Preparation

```bash
# Extract images from JSON files
python data_preprocess/data_preprocess.py --task extract --json-dir /path/to/json/files --save-dir /path/to/images

# Convert JSON annotations to YOLO format
python data_preprocess/data_preprocess.py --task convert --json-dir /path/to/json/files --save-dir /path/to/labels

# Process and interpolate keypoints
python data_preprocess/data_preprocess.py --task process --json-dir /path/to/json/files --save-dir /path/to/processed

# Or run all tasks at once
python data_preprocess/data_preprocess.py --task all --json-dir /path/to/json/files --save-dir /path/to/output --extract-dir /path/to/images --convert-dir /path/to/labels --process-dir /path/to/processed
```

## Download Weights
Download the PPF-YOLO.pt weights file from the [Releases](https://github.com/Akacaesarp/PPF-YOLO-Pretrained-Pose-Detection-in-Field-YOLO/releases)

## Install

```bash
# Clone
git clone https://github.com/Akacaesarp/PPF-YOLO-Pretrained-Pose-Detection-in-Field-YOLO.git
cd PPF-YOLO-Pretrained-Pose-Detection-in-Field-YOLO

# Install requirements
pip install -r requirements.txt

```
## Dataset Format
```bash
maize pose dataset
├── images
│   ├── train
│   ├── val
│   └── test
└── labels
│   ├── train
│   ├── val
│   └── test
├── Maize_pose.yaml
```

## Training
To train the model, simply run train.py:
```bash
# Run training using train.py
python train.py
```
The train.py script contains:
```bash
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('PPF-YOLO.yaml')
    model.train(data="Maize_pose.yaml",
                imgsz=1280,
                task='pose',
                epochs=120,
                batch=2)
```
## Inference
We provide sample images in the test_images folder that you can use to test the model. To run inference:
```bash
# Run inference using Detect.py
python Detect.py
```
The Detect.py script contains:
```bash
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('PPF-YOLO.pt')
    result=model.predict(R"test_images", 
                save = True,
                show_boxes=False,
                show_labels=False,
                )
```
