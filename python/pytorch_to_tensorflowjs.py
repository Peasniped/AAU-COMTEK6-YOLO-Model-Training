from    ultralytics import YOLO
"""
Virker umiddelbart ikke på Windows.
Kræver 'tensorflow==2.10' som jeg ikke kan installere med PIP
"""

def convert_yolo_to_tfjs(pt_model_path: str, output_dir: str = "tfjs_model") -> None:
    model = YOLO(pt_model_path)
    model.export(format="tfjs")  # creates '/yolo11n_web_model'


if __name__ == "__main__":
    # Example usage
    convert_yolo_to_tfjs("runs/horse_features/weights/best.pt")

    # Virker ikke på Windows