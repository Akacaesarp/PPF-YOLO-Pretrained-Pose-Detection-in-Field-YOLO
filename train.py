import warnings

# 禁用所有警告
warnings.filterwarnings("ignore")

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'PPF-YOLO.yaml')
    model.train(data=r"dataset\maize-pose.yaml",  # your dataset path
                imgsz=1280,
                task='pose',
                epochs=120,
                batch=2,
                )