import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ultralytics import YOLO

N_SAMPLES = 5
CLASS_NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]


def main():
    print("=" * 60)
    print("Projeto 3 — Inferência com model.tflite (Edge AI)")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.tflite")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model.tflite não encontrado em {model_path}. "
            "Rode optimize_model.py antes de executar a inferência."
        )

    val_dir = os.path.join(script_dir, "dataset", "images", "val")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Pasta de validação não encontrada: {val_dir}")

    all_images = sorted(
        [f for f in os.listdir(val_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    if len(all_images) < N_SAMPLES:
        raise ValueError(
            f"Esperava pelo menos {N_SAMPLES} imagens em {val_dir}, encontrou {len(all_images)}."
        )
    
    # carrega especificamente o .tflite (artefato de edge), não o .pt
    model = YOLO(model_path, task="detect")

    sample_images = all_images[:N_SAMPLES]
    sample_paths = [os.path.join(val_dir, f) for f in sample_images]

    print(f"\nRodando inferência em {len(sample_paths)} amostras usando model.tflite:\n")
    print(f"{'Imagem':<35} {'Detecções':>10}  Detalhes")
    print("-" * 70)

    total_detections = 0

    for path in sample_paths:
        # uma imagem por vez (batch=1) — restrição do model.tflite exportado,
        # e também o cenário real de uso em edge
        result = model.predict(
            source=path,
            imgsz=640,
            save=True,
            project=os.path.join(script_dir, "runs", "detect"),
            name="inferencia_exemplos/predicoes",
            exist_ok=True,
            verbose=False,
        )[0]

        n_det = len(result.boxes)
        total_detections += n_det

        if n_det > 0:
            class_ids = result.boxes.cls.tolist()
            class_counts = {}
            for cid in class_ids:
                cname = CLASS_NAMES[int(cid)] if int(cid) < len(CLASS_NAMES) else str(int(cid))
                class_counts[cname] = class_counts.get(cname, 0) + 1
            details = ", ".join(f"{v}x {k}" for k, v in class_counts.items())
        else:
            details = "nenhuma detecção"

        print(f"{os.path.basename(path):<35} {n_det:>10}  [{details}]")

    print("-" * 70)
    print(f"{'TOTAL':<35} {total_detections:>10}")
    print(f"\n✅ Imagens anotadas salvas em: runs/detect/inferencia_exemplos/predicoes/")
    print("   (Abra essa pasta para verificar visualmente as bounding boxes preditas)")


if __name__ == "__main__":
    main()