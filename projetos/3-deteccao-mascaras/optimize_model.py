import os
os.environ["YOLO_AUTOINSTALL"] = "False"

import shutil
from ultralytics import YOLO

def main():
    if not os.path.exists("model.pt"):
        raise FileNotFoundError("model.pt não encontrado. Rode train_model.py antes.")

    model = YOLO("model.pt")
    exported_path = model.export(format="tflite", imgsz=640)
    # usa o caminho retornado pela própria função, em vez de um caminho fixo,
    # pois o nome/local do arquivo exportado pode variar entre versões da Ultralytics
    shutil.copy(exported_path, "model.tflite")

    size_mb = os.path.getsize("model.tflite") / (1024 * 1024)
    print(f"Modelo exportado com sucesso: {exported_path} -> model.tflite")
    print(f"Tamanho final de model.tflite: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
