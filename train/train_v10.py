import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLOv10
 
if __name__ == '__main__':
    model = YOLOv10('ultralytics/cfg/models/v10/yolov10n-DualConv.yaml')
    #model.load('yolov10n.pt') # loading pretrain weights
    model.train(data='data/NEU-DET.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=4,
                close_mosaic=10,
                device='0',
                optimizer='SGD', # using SGD
                project='runs/train',
                name='exp',
                )
 
 