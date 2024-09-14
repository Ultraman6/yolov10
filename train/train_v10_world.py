from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

data = dict(
    train=dict(
        yolo_data=["Objects365.yaml"],
        grounding_data=[
            dict(
                img_path="../datasets/flickr30k/images",
                json_file="../datasets/flickr30k/final_flickr_separateGT_train.json",
            ),
            dict(
                img_path="../datasets/GQA/images",
                json_file="../datasets/GQA/final_mixed_train_no_coco.json",
            ),
        ],
    ),
    val=dict(yolo_data=["lvis.yaml"]),
)
model = YOLOWorld("yolov8s-worldv2.yaml")
model.train(data=data, batch=128, epochs=100, trainer=WorldTrainerFromScratch)