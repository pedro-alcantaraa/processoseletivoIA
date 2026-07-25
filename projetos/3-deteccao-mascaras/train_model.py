import os
import shutil
from ultralytics import YOLO

EPOCHS = 20 
IMG_SIZE = 640
BATCH_SIZE = 8

def main():
    if not os.path.exists("dataset/data.yaml"):
        raise FileNotFoundError("dataset/data.yaml não encontrado.")

    model = YOLO("yolo11n.pt")
    results = model.train(
        data="dataset/data.yaml",
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device="cpu",
    )
    shutil.copy(results.save_dir / "weights" / "best.pt", "model.pt")
    print("Treinamento concluído. model.pt gerado com sucesso.")

if __name__ == "__main__":
    main()