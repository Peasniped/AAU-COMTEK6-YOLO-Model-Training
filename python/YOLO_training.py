import os
import yaml
import torch
import torchvision
from ultralytics import YOLO

def _update_yaml_paths(yaml_path: str, project_root: str | None = None) -> None:
    project_root = os.getcwd() if project_root is None else project_root
    
    # Load existing YAML
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    # Build absolute paths
    abs_train = os.path.abspath(os.path.join(project_root, data['train']))
    abs_val = os.path.abspath(os.path.join(project_root, data['val']))

    # Update YAML
    data['train'] = abs_train
    data['val'] = abs_val

    # Save it back
    with open(yaml_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

    print(f"Updated paths:\n  train: {abs_train}\n  val: {abs_val}")

def _get_device() -> str:
    cuda = torch.cuda.is_available()
    if cuda:
        print("Torchvision test:", torchvision.ops.nms)  # Should not error
        print("Torch cuda is available:", torch.cuda.is_available())        # should be True
        print("Torch cuda device name", torch.cuda.get_device_name(0))    # “NVIDIA GeForce RTX 2080 SUPER”
        print("Torch cuda device VRAM", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB\n")
        return "cuda"
    else:
        print("CUDA device not detected")
        return "cpu"

def _get_worker_count(scalar:float = 3/4) -> int:
    cpus = os.cpu_count()
    workers = int(cpus * scalar)
    print(f"CPU's available is {cpus}, using {workers} workers (scalar {scalar})")
    return workers

def _get_batch(device: str, imgz: int) -> tuple[int, int]:
    if device == "cuda":
        if imgz == 1280:
            batch = 4
        elif imgz == 960:
            batch = 8
        elif imgz == 640:
            batch = 16

    else:
        imgz = 640
        batch = 4

    print(f"Using image size: {imgz}^2 with batch size: {batch}")

    return imgz, batch

def train(epochs:int, imgz:int, base_model = "yolo11", model_skew: str = "nano") -> None:
    model_skews = ["nano", "small", "medium"]

    if not model_skew in model_skews:
        raise SyntaxError(f"paramter 'model_skew' must have one of following values: {model_skews}")
    
    # Load a pre-trained YOLOv8 model (nano, small, medium, etc.)
    model = YOLO(f"models/{base_model}{model_skew[0]}.pt")
    
    _update_yaml_paths(yaml_path='dataset/horse_data.yaml')

    device = _get_device()
    imgz, batch = _get_batch(device, imgz)
    workers = _get_worker_count()
    patience_scalar = 0.2
    patience = int(epochs * patience_scalar)
    print(f"Using patience of {patience} epochs which is {patience_scalar*100}% of total epoch count {epochs}")
    print("Starting training module...\n\n")

    model.train(
        name        = "horse_features",
        project     = "runs",
        data        = "dataset/horse_data.yaml",
        exist_ok    = True,              # Allows overwriting of files
        epochs      = epochs,
        patience    = patience, # Stops after n epochs of no improvement
        imgsz       = imgz,
        batch       = batch,
        workers     = workers,
        device      = device
    )

    os.remove("yolo11n.pt")
    print("Removed temporary model from root 'yolo11n.pt' used by YOLO for running Automatic Mixed Precision (AMP) checks!")

if __name__ == "__main__":
    train(200, 960, "yolo11", "nano")