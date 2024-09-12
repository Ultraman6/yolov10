import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# 加载模型
model = YOLO("yolov10s.pt")
names = model.names

# 图片文件路径
image_path = "asset/img_2.png"
image = cv2.imread(image_path)
assert image is not None, "Error reading image file"

# 输出目录
crop_dir_name = "ultralytics_crop"
if not os.path.exists(crop_dir_name):
    os.mkdir(crop_dir_name)

# 运行模型预测
results = model.predict(image, show=False)
boxes = results[0].boxes.xyxy.cpu().tolist()
clss = results[0].boxes.cls.cpu().tolist()
annotator = Annotator(image, line_width=2, example=names)

# 如果有检测到的物体，绘制边框并裁剪
if boxes:
    for idx, (box, cls) in enumerate(zip(boxes, clss), start=1):
        annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

        # 裁剪物体
        crop_obj = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        cv2.imwrite(os.path.join(crop_dir_name, f"{idx}.png"), crop_obj)

# 显示图像
cv2.imshow("ultralytics", image)
cv2.waitKey(0)  # 等待按键以继续
cv2.destroyAllWindows()
