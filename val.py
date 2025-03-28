import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R"PPF-YOLO.pt")
    results = model.val(data=r"dataset\maize-pose.yaml",
                        split='test',
                        imgsz=1280,
                        save_txt=True,
                        plots=False)
