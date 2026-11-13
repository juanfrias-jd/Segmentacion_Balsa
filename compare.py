"""
Tabla comparativa Taller 2 vs. Proyecto (Parte 2 del enunciado).

  python compare.py

Lee results/metrics_*.json y los resultados originales del Taller 2 (config.yaml)
y genera:
  results/tabla_comparativa.csv / .md
  results/figures/comparacion_metricas.png
  results/datos_para_analisis.md   (diferencias numéricas; la INTERPRETACIÓN
                                    la escriben los estudiantes)

La columna "Análisis" se deja para que el grupo la redacte: el enunciado
penaliza conclusiones escritas con generadores de texto.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils.config import load_config, resolve  # noqa: E402

ORDER = [("zeroshot", "Taller 2: Mask2Former COCO sin re-entrenar"),
         ("maskrcnn", "Taller 2: Mask R-CNN re-entrenado en balsa"),
         ("mask2former", "Proyecto: Mask2Former panóptico balsa")]
PLACEHOLDER = "[Escribir análisis del grupo]"


def fmt(v, nd=3):
    return "N/A" if v is None or pd.isna(v) else f"{v:.{nd}f}"


def main():
    cfg = load_config()
    th = cfg["evaluation"]["thresholds"]
    t2 = cfg["taller2_original"]
    rows = [{
        "Modelo": t2["modelo"], "Evaluado en": t2["dataset"] + " (resultado ORIGINAL del taller)",
        "IoU": t2["IoU"], "mAP": t2["mAP"], "F1": t2["F1"], "Precisión": t2["Precision"],
        "Recall": None, "PQ": None, "mAP50": None, "FPS": None, "Stuff": "No (solo instancias)",
        "Etiquetas compatibles": "No (Área Caminable, Construida, Zona Verde, Carretera)",
    }]
    found = {}
    for key, label in ORDER:
        p = resolve(f"results/metrics_{key}.json")
        if not p.exists():
            print(f"[omitido] {p.name} no existe (ejecute evaluate.py --model {key})")
            continue
        m = json.load(open(p, encoding="utf-8"))
        found[key] = m
        rows.append({
            "Modelo": label, "Evaluado en": f"Test madera balsa ({m['n_images']} imágenes, {m['n_gt_balsa']} piezas)",
            "IoU": m.get("IoU"), "mAP": m.get("mAP"), "F1": m.get("F1"), "Precisión": m.get("Precision"),
            "Recall": m.get("Recall"), "PQ": m.get("PQ"), "mAP50": m.get("mAP50"), "FPS": m.get("FPS_inferencia"),
            "Stuff": "Sí (mesa, fondo)" if m.get("tiene_stuff") else "No (solo instancias)",
            "Etiquetas compatibles": {"zeroshot": "No: clases COCO mapeadas a balsa/mesa/fondo",
                                      "maskrcnn": "Sí (balsa)", "mask2former": "Sí (balsa, mesa, fondo)"}[key],
        })
    df = pd.DataFrame(rows)

    def cumple(r):
        vals = [(k, r[k]) for k in ["IoU", "mAP", "F1"]]
        if any(v is None or pd.isna(v) for _, v in vals):
            return "N/A"
        bad = [k for k, v in vals if v < th[k]]
        return "Sí" if not bad else "No (" + ", ".join(bad) + ")"
    df["Cumple mínimos"] = df.apply(cumple, axis=1)
    df["Análisis"] = PLACEHOLDER

    out = resolve("results")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "tabla_comparativa.csv", index=False, encoding="utf-8-sig")
    show = df.copy()
    for c in ["IoU", "mAP", "F1", "Precisión", "Recall", "PQ", "mAP50"]:
        show[c] = show[c].apply(fmt)
    show["FPS"] = show["FPS"].apply(lambda v: fmt(v, 1))
    with open(out / "tabla_comparativa.md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(show.columns) + " |\n|" + "---|" * len(show.columns) + "\n")
        for _, r in show.iterrows():
            f.write("| " + " | ".join(str(x) for x in r.values) + " |\n")
    print(show[["Modelo", "IoU", "mAP", "F1", "Precisión", "Cumple mínimos"]].to_string(index=False))

    # ---- Gráfica de barras ----
    metrics = ["IoU", "mAP", "F1", "Precisión"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    n = len(df)
    w = 0.8 / n
    cmap = {t2["modelo"]: "#9ca3af", ORDER[0][1]: "#a855f7", ORDER[1][1]: "#f59e0b", ORDER[2][1]: "#16a34a"}
    for i, (_, r) in enumerate(df.iterrows()):
        vals = [0.0 if r[m] is None or pd.isna(r[m]) else float(r[m]) for m in metrics]
        x = np.arange(len(metrics)) + (i - (n - 1) / 2) * w
        lab = r["Modelo"] + (" [otro dataset]" if i == 0 else "")
        bars = ax.bar(x, vals, w, label=lab, color=cmap.get(r["Modelo"]))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    for k, xi in zip(["IoU", "mAP", "F1"], range(3)):
        ax.hlines(th[k], xi - 0.45, xi + 0.45, colors="red", linestyles="--", lw=1.2)
    ax.set_xticks(range(len(metrics)), metrics)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Valor")
    ax.set_title("Comparación de métricas (línea roja: mínimo exigido)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out / "figures" / "comparacion_metricas.png", dpi=150); plt.close()

    # ---- Datos numéricos de apoyo para escribir el análisis ----
    lines = ["# Datos para el análisis (los escribe el grupo)\n",
             "Diferencias calculadas automáticamente. La interpretación debe redactarla el grupo.\n"]
    if "mask2former" in found:
        p = found["mask2former"]
        for key, label in ORDER[:2]:
            if key in found:
                b = found[key]
                lines.append(f"\n## Proyecto vs. {label}")
                for k in ["IoU_balsa", "mAP", "mAP50", "F1", "Precision", "Recall", "PQ_things", "FPS_inferencia"]:
                    if p.get(k) is not None and b.get(k) is not None:
                        lines.append(f"- {k}: {b[k]:.3f} -> {p[k]:.3f} (Δ {p[k]-b[k]:+.3f})")
                lines.append(f"- TP/FP/FN: {b['TP']}/{b['FP']}/{b['FN']} -> {p['TP']}/{p['FP']}/{p['FN']}")
        lines.append("\n## IoU por clase del proyecto")
        for c, v in p["IoU_por_clase"].items():
            lines.append(f"- {cfg['id2name'][int(c)]}: {fmt(v)}")
        if "MAE_largo_px" in p:
            lines.append(f"\n## Error de medición (piezas emparejadas)\n- largo: {p['MAE_largo_px']:.1f} px "
                         f"({p['MAPE_largo_%']:.1f} %)\n- ancho: {p['MAE_ancho_px']:.1f} px ({p['MAPE_ancho_%']:.1f} %)")
    lines += ["\n## Preguntas guía para la columna de análisis",
              "- ¿Por qué el modelo del Taller 2 sin re-entrenar no reconoce 'madera balsa'? (clases COCO)",
              "- ¿Qué aporta tener 'stuff' (mesa/fondo) frente a un modelo solo de instancias?",
              "- ¿En qué casos falla cada modelo? Revise results/examples/ (piezas tocándose, sombras, piezas delgadas).",
              "- ¿Qué métrica es más exigente y por qué (mAP 0.50:0.95 vs. mAP50)?",
              "- ¿Cómo influyen la cantidad de imágenes, el aumento de datos y el tiempo de inferencia?",
              "- ¿Por qué los resultados originales del Taller 2 no son directamente comparables?"]
    with open(out / "datos_para_analisis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nTabla -> {out/'tabla_comparativa.md'}\nGráfica -> {out/'figures'/'comparacion_metricas.png'}")


if __name__ == "__main__":
    main()
