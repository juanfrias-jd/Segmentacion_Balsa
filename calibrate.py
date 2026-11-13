"""
Calibración píxel -> centímetro (puntos extra: medidas en cm).

1) Generar e imprimir el marcador ArUco:
     python calibrate.py --generate
   Crea results/aruco_marker_5cm.png (300 DPI). Imprímalo al 100 % (sin
   "ajustar a la página") y MIDA el lado del cuadrado negro con una regla.
   Si no mide 5.0 cm, escriba el valor real en configs/config.yaml
   (measure.aruco_size_cm).

2) Calibrar con la cámara FIJA (trípode/soporte, mirando la mesa desde arriba):
     python calibrate.py                 # ESPACIO o C guarda cuando ve el marcador
   Se guarda configs/calibration.json con la homografía imagen -> plano mesa.
   Después de calibrar puede retirar el marcador si la cámara no se mueve.

3) Alternativa sin marcador (escala simple con una regla):
     python calibrate.py --manual 10     # clic en dos puntos separados 10 cm

Por qué homografía: el marcador da 4 esquinas con coordenadas reales
conocidas (0,0), (L,0), (L,L), (0,L). Con ellas se calcula la transformación
proyectiva que corrige la perspectiva si la cámara está algo inclinada; luego
cada contorno de pieza se proyecta al plano de la mesa y se mide en cm.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from utils.config import load_config, resolve
from utils.io import imwrite
from utils.measure import calibrate_from_aruco, calibrate_from_points, detect_aruco


def generate(cfg, dpi=300):
    mc = cfg["measure"]
    size_cm = mc["aruco_size_cm"]
    px = int(round(size_cm / 2.54 * dpi))
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, mc["aruco_dict"]))
    marker = cv2.aruco.generateImageMarker(d, mc["aruco_id"], px)
    margin = int(1.0 / 2.54 * dpi)                      # 1 cm de margen blanco
    page = np.full((px + 2 * margin + 120, px + 2 * margin), 255, np.uint8)
    page[margin:margin + px, margin:margin + px] = marker
    cv2.putText(page, f"ArUco {mc['aruco_dict']} id={mc['aruco_id']}  lado = {size_cm} cm (imprimir al 100%)",
                (10, px + 2 * margin + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    out = resolve(f"results/aruco_marker_{size_cm:g}cm.png")
    imwrite(out, page)
    try:  # agrega la resolución (DPI) para que se imprima al tamaño correcto
        from PIL import Image
        Image.open(out).save(out, dpi=(dpi, dpi))
    except Exception:
        pass
    print(f"Marcador guardado en {out}. Imprímalo al 100 % y verifique que mida {size_cm} cm.")


def live(cfg, camera):
    mc = cfg["measure"]
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["realtime"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["realtime"]["height"])
    print("Muestre el marcador sobre la mesa. ESPACIO/C = guardar calibración, Q = salir")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()
        corners = detect_aruco(frame, mc["aruco_dict"], mc["aruco_id"])
        if corners is not None:
            cv2.polylines(view, [corners.astype(np.int32)], True, (0, 255, 0), 2)
            calib, _ = calibrate_from_aruco(frame, mc["aruco_size_cm"], mc["aruco_dict"], mc["aruco_id"])
            cv2.putText(view, f"{calib.px_per_cm:.2f} px/cm  [ESPACIO] guardar", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(view, "Marcador no detectado", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Calibracion ArUco", view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord(" "), ord("c")) and corners is not None:
            calib.save(resolve(mc["calibration_file"]))
            print(f"Guardado {mc['calibration_file']}: {calib.source}, {calib.px_per_cm:.2f} px/cm")
            break
        if k in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def manual(cfg, camera, distance_cm):
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["realtime"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["realtime"]["height"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("No se pudo leer la cámara")
    pts = []

    def on_click(e, x, y, *_):
        if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
    cv2.namedWindow("Regla"); cv2.setMouseCallback("Regla", on_click)
    print(f"Haga clic en dos puntos separados {distance_cm} cm (p.ej. marcas de una regla). Q = cancelar")
    while len(pts) < 2:
        v = frame.copy()
        for p in pts:
            cv2.circle(v, p, 4, (0, 0, 255), -1)
        cv2.imshow("Regla", v)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            return
    cv2.destroyAllWindows()
    calib = calibrate_from_points(pts[0], pts[1], distance_cm, [frame.shape[1], frame.shape[0]])
    calib.save(resolve(cfg["measure"]["calibration_file"]))
    print(f"Guardado: {calib.source} -> {calib.px_per_cm:.2f} px/cm")


if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="crear el marcador para imprimir")
    ap.add_argument("--manual", type=float, metavar="CM", help="calibrar con dos clics a CM de distancia")
    ap.add_argument("--camera", type=int, default=cfg["realtime"]["camera"])
    a = ap.parse_args()
    if a.generate:
        generate(cfg)
    elif a.manual:
        manual(cfg, a.camera, a.manual)
    else:
        live(cfg, a.camera)
