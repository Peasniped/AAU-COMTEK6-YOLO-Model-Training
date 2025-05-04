import cv2
import os
from ultralytics import YOLO

# ENV YOLO_CONFIG_DIR 

# Load your best model
model = YOLO('runs\\horse_features\\weights\\best.pt')

def detect_horse_features(model: YOLO, image_path: str) -> tuple:
    # Run model on image
    results = model(image_path, imgsz=640, conf=0.56)
    objects = []

    # Load image to draw bbox
    img = cv2.imread(image_path)

    for obj in results:
        for bbox, classs_id, confidence in zip(obj.boxes.xyxy, obj.boxes.cls, obj.boxes.conf):
            label = model.names[int(classs_id)]
            if label in ["ear", "eye", "nostril", "head", "mouth"]:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                text = f'{label} {confidence:.2f}'
                cv2.putText(img, text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                objects.append({"label": label, "bbox": bbox, "conficence": confidence})
    
    return img, objects

if __name__ == "__main__":

    for file in ["dataset/test/" + file for file in os.listdir("dataset/test") if file.endswith(".jpg")]:
        print(file)
        img, _ = detect_horse_features(model, file)

        # Resize if needed
        # Define monitor size
        max_width = 1600
        max_height = 800

        height, width = img.shape[:2]
        if width > max_width or height > max_height:
            scale_w = max_width / width
            scale_h = max_height / height
            scale = min(scale_w, scale_h)
            new_w = int(width * scale)
            new_h = int(height * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Show image
        cv2.imshow(f'Detections on image {file}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()