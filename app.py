"""
Interfaz WEB de la demostración en tiempo real (Flask + HTML/CSS/JS).

  python app.py                       # abre http://127.0.0.1:5000
  python app.py --model maskrcnn      # línea base del Taller 2
  python app.py --camera video.mp4    # con un video en lugar de la cámara

La página muestra el video segmentado (things + stuff), la tabla de piezas
con largo y ancho (px o cm), el porcentaje de área de cada región, botones
de calibración/captura y las métricas obtenidas en el conjunto de prueba.
"""
from __future__ import annotations

import argparse
import json
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template, send_from_directory

from utils.config import get_device, load_config, resolve
from utils.measure import Calibration
from utils.pipeline import CameraStream, RealtimePipeline, black_frame, load_predictor


class Worker:
    """Hilo que lee la cámara, ejecuta el modelo y guarda el último resultado."""

    def __init__(self, pipe: RealtimePipeline, cam: CameraStream):
        self.pipe, self.cam = pipe, cam
        self.jpeg = cv2.imencode(".jpg", black_frame("Iniciando camara..."))[1].tobytes()
        self.raw = None
        self.vis = None
        self.lock = threading.Lock()
        self.pending = []           # acciones pedidas desde la web
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            ok, frame = self.cam.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            vis, _ = self.pipe.process(frame)
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self.lock:
                self.raw, self.vis, self.jpeg = frame, vis, buf.tobytes()

    def frames(self):
        last = None
        while True:
            with self.lock:
                j = self.jpeg
            if j is not last:
                last = j
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
            time.sleep(0.01)


def load_metrics(cfg):
    """Métricas de test (results/metrics_*.json) para mostrarlas en la página."""
    names = {"mask2former": "Proyecto: Mask2Former panóptico", "maskrcnn": "Taller 2: Mask R-CNN re-entrenado",
             "zeroshot": "Taller 2: Mask2Former COCO sin re-entrenar"}
    rows = []
    t2 = cfg["taller2_original"]
    rows.append({"modelo": "Taller 2 original (dataset cartografía)", "IoU": t2["IoU"], "mAP": t2["mAP"],
                 "F1": t2["F1"], "Precision": t2["Precision"], "PQ": None, "FPS": None})
    for k, n in names.items():
        p = resolve(f"results/metrics_{k}.json")
        if p.exists():
            m = json.load(open(p, encoding="utf-8"))
            rows.append({"modelo": n, "IoU": m.get("IoU"), "mAP": m.get("mAP"), "F1": m.get("F1"),
                         "Precision": m.get("Precision"), "PQ": m.get("PQ"), "FPS": m.get("FPS_inferencia")})
    return {"rows": rows, "umbrales": cfg["evaluation"]["thresholds"]}


def create_app(worker: Worker, cfg) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    pipe = worker.pipe

    @app.route("/")
    def index():
        return render_template("index.html", modelo=getattr(pipe.pred, "name", ""),
                               clases=[c for c in cfg["classes"]])

    @app.route("/video_feed")
    def video_feed():
        return Response(worker.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/stats")
    def stats():
        return jsonify(pipe.last)

    @app.route("/api/metrics")
    def metrics():
        return jsonify(load_metrics(cfg))

    @app.route("/api/figures")
    def figures():
        d = resolve("results/figures")
        want = ["perdida_mask2former.png", "perdida_maskrcnn.png", "comparacion_metricas.png",
                "metricas_val_mask2former.png", "dataset_muestras.png"]
        return jsonify([f for f in want if (d / f).exists()])

    @app.route("/figures/<path:name>")
    def figure(name):
        return send_from_directory(resolve("results/figures"), name)

    @app.route("/api/calibrate", methods=["POST"])
    def calibrate():
        with worker.lock:
            frame = None if worker.raw is None else worker.raw.copy()
        if frame is None:
            return jsonify(ok=False, msg="Sin imagen de la cámara")
        ok, msg = pipe.calibrate(frame)
        return jsonify(ok=ok, msg=msg)

    @app.route("/api/reset_calibration", methods=["POST"])
    def reset_cal():
        pipe.calib = Calibration()
        pipe.tracker.tracks.clear()
        return jsonify(ok=True, msg="Medidas en píxeles")

    @app.route("/api/snapshot", methods=["POST"])
    def snapshot():
        with worker.lock:
            vis = None if worker.vis is None else worker.vis.copy()
        if vis is None:
            return jsonify(ok=False, msg="Sin imagen")
        p = pipe.save_snapshot(vis, resolve(cfg["realtime"]["results_dir"]))
        return jsonify(ok=True, msg=f"Guardado {p.name} (+ CSV de medidas)")

    @app.route("/api/toggle/<what>", methods=["POST"])
    def toggle(what):
        if what == "stuff":
            pipe.show_stuff = not pipe.show_stuff
            return jsonify(ok=True, value=pipe.show_stuff)
        if what == "masks":
            pipe.show_masks = not pipe.show_masks
            return jsonify(ok=True, value=pipe.show_masks)
        if what == "aruco":
            pipe.aruco_live = not pipe.aruco_live
            return jsonify(ok=True, value=pipe.aruco_live)
        return jsonify(ok=False), 404

    return app


def main():
    cfg = load_config()
    rc = cfg["realtime"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mask2former", choices=["mask2former", "maskrcnn", "zeroshot"])
    ap.add_argument("--weights")
    ap.add_argument("--camera", default=str(rc["camera"]))
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    device = get_device("cpu" if args.cpu else cfg["project"]["device"])
    print(f"Cargando modelo '{args.model}' en {device}...")
    pred = load_predictor(args.model, cfg, device, args.weights)
    pipe = RealtimePipeline(pred, cfg, Calibration.load(resolve(cfg["measure"]["calibration_file"])))
    src = int(args.camera) if args.camera.isdigit() else args.camera
    worker = Worker(pipe, CameraStream(src, rc["width"], rc["height"]))
    print(f"Abra http://127.0.0.1:{args.port} en el navegador")
    create_app(worker, cfg).run(host="0.0.0.0", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
