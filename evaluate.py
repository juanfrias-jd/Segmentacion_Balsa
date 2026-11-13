"""
Evaluación sobre el MISMO conjunto de prueba de madera balsa (data/processed/test).

  python evaluate.py --model mask2former   # modelo del proyecto (fine-tuning)
  python evaluate.py --model maskrcnn      # arquitectura Taller 2 re-entrenada en balsa
  python evaluate.py --model zeroshot      # Mask2Former COCO del Taller 2 SIN re-entrenar
  python evaluate.py --all                 # los tres (los que tengan pesos)

Guarda results/metrics_<modelo>.json y ejemplos visuales en
results/examples/<modelo>/ (anotación real vs. predicción).
Luego ejecute:  python compare.py   para la tabla comparativa.
"""
from __future__ import annotations

import argparse
import json

from data.dataset import BalsaDataset
from utils.config import get_device, load_config, resolve
from utils.evaluation import evaluate_predictor

KEYS = ["IoU", "mIoU", "IoU_balsa", "mAP", "mAP50", "mAP75", "F1", "Precision", "Recall",
        "PQ", "PQ_things", "PQ_stuff", "MAE_largo_px", "MAE_ancho_px", "FPS_inferencia"]


def build_predictor(name, cfg, device, weights=None):
    if name == "mask2former":
        from models.mask2former_panoptic import Mask2FormerPredictor
        return Mask2FormerPredictor(cfg, weights=weights or resolve(cfg["mask2former"]["weights"]),
                                    device=device), True
    if name == "maskrcnn":
        from models.maskrcnn_baseline import MaskRCNNPanopticPredictor
        return MaskRCNNPanopticPredictor(cfg, weights=weights or resolve(cfg["maskrcnn"]["weights"]),
                                         device=device), False
    if name == "zeroshot":
        from models.mask2former_panoptic import Mask2FormerPredictor
        return Mask2FormerPredictor(cfg, device=device, zeroshot=True,
                                    infer_size=cfg["data"]["input_size"]), True
    raise ValueError(name)


def run(name, cfg, device, split, weights=None, data_root=None):
    ds = BalsaDataset(resolve(data_root or cfg["data"]["processed_dir"]) / split, tuple(cfg["data"]["input_size"]))
    pred, has_stuff = build_predictor(name, cfg, device, weights)
    print(f"\n=== {pred.name} | {split}: {len(ds)} imágenes ===")
    res = evaluate_predictor(pred, ds, cfg, has_stuff=has_stuff,
                             examples_dir=resolve(f"results/examples/{name}"), n_examples=8, progress=True)
    res.update(modelo=pred.name, clave=name, particion=split, tiene_stuff=has_stuff,
               nombres_clase=cfg["id2name"])
    out = resolve(f"results/metrics_{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    th = cfg["evaluation"]["thresholds"]
    for k in KEYS:
        v = res.get(k)
        if v is None:
            print(f"  {k:16s}   N/A")
            continue
        flag = ""
        if k in th:
            flag = "  OK" if v >= th[k] else f"  (< {th[k]})"
        print(f"  {k:16s} {v:8.4f}{flag}")
    print(f"  -> {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mask2former", "maskrcnn", "zeroshot"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--weights")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data", help="carpeta procesada (por defecto data/processed)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    device = get_device("cpu" if args.cpu else cfg["project"]["device"])

    names = [args.model] if args.model else []
    if args.all:
        names = []
        for n in ["mask2former", "maskrcnn"]:
            if resolve(cfg[n]["weights"]).exists():
                names.append(n)
            else:
                print(f"[omitido] {n}: no existe {cfg[n]['weights']}")
        names.append("zeroshot")
    if not names:
        ap.error("indique --model o --all")
    for n in names:
        run(n, cfg, device, args.split, args.weights if len(names) == 1 else None, args.data)


if __name__ == "__main__":
    main()
