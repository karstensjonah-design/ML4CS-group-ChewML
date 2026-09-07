"""Smoke-Test: spielt eine echte Aufnahme als Sensorstream in die laufende App."""
import json, sys, time, urllib.request, zipfile
from pathlib import Path

import pandas as pd

URL   = "http://127.0.0.1:8000/data"
KEYS  = ["accelerationX", "accelerationY", "accelerationZ",
         "rotationRateX", "rotationRateY", "rotationRateZ",
         "pitch", "roll", "yaw"]


def post(rows):
    body = json.dumps({"payload": rows}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status


def replay(zip_path, secs=14.0, chunk=0.5, speed=0.12):
    z = zipfile.ZipFile(zip_path)
    name = [n for n in z.namelist()
            if n.endswith(".csv") and n not in {"Metadata.csv", "Annotation.csv"}][0]
    df = pd.read_csv(z.open(name))
    t0 = df["seconds_elapsed"].iloc[0] + 5.0          # 5 s Anlauf ueberspringen
    df = df[(df["seconds_elapsed"] >= t0) & (df["seconds_elapsed"] < t0 + secs)]

    base_ns = time.time_ns()
    start_s = df["seconds_elapsed"].iloc[0]
    batch, sent, t_next = [], 0, chunk
    for _, r in df.iterrows():
        rel = float(r["seconds_elapsed"]) - start_s
        batch.append({"time": base_ns + int(rel * 1e9),
                      "values": {k: float(r[k]) for k in KEYS}})
        if rel >= t_next:
            post(batch); sent += len(batch); batch = []; t_next += chunk
            time.sleep(chunk * speed)                  # speed=1.0 -> Echtzeit
    if batch:
        post(batch); sent += len(batch)
    return sent


if __name__ == "__main__":
    p = Path(sys.argv[1])
    n = replay(p, secs=float(sys.argv[2]) if len(sys.argv) > 2 else 14.0,
               speed=float(sys.argv[3]) if len(sys.argv) > 3 else 0.12)
    print(f"{p.name}: {n} Samples gesendet")
