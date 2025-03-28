import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R"PPF-YOLO.pt")
    model.predict(source=R'test_images',
                  imgsz=1280,
                  save = True,
                  show_labels=False,
                  show_boxes=False,
                )