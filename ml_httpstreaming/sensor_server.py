"""
HTTP server for Sensor Logger app (iOS/Android).
Listens on 0.0.0.0:8000, accepts POST /data, prints and saves to CSV.
New CSV file per server start. Shows live Hz per sensor.
"""

import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

# New timestamped file each run
_start_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILE = f"sensor_data_{_start_ts}.csv"
CSV_HEADERS_WRITTEN = set()

# Hz tracking: sensor -> list of recent arrival times (seconds)
_arrival_times: dict[str, list] = defaultdict(list)
_HZ_WINDOW = 5.0  # calculate Hz over last 5 seconds


def hz_for(sensor: str) -> float:
    now = time.monotonic()
    times = _arrival_times[sensor]
    times.append(now)
    # Drop entries older than the window
    cutoff = now - _HZ_WINDOW
    while times and times[0] < cutoff:
        times.pop(0)
    elapsed = times[-1] - times[0] if len(times) > 1 else _HZ_WINDOW
    return (len(times) - 1) / elapsed if elapsed > 0 else 0.0


def nanoseconds_to_iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()


def flatten_values(values: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in values.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_values(v, key))
        else:
            flat[key] = v
    return flat


def write_csv_row(row: dict) -> None:
    fieldnames = list(row.keys())
    header_key = tuple(fieldnames)
    needs_header = header_key not in CSV_HEADERS_WRITTEN

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
            CSV_HEADERS_WRITTEN.add(header_key)
        writer.writerow(row)


def process_payload(payload: list, received_at: str) -> None:
    for entry in payload:
        sensor_name = entry.get("name", "unknown")
        sensor_time_ns = entry.get("time", 0)
        values = entry.get("values", {})

        sensor_time_iso = nanoseconds_to_iso(sensor_time_ns) if sensor_time_ns else received_at
        flat = flatten_values(values)
        hz = hz_for(sensor_name)

        values_str = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in flat.items()
        )
        print(f"[{sensor_time_iso}] {sensor_name:<22} {hz:5.1f} Hz  {values_str}")

        row = {"received_at": received_at, "sensor_time": sensor_time_iso, "sensor": sensor_name, **flat}
        write_csv_row(row)


class SensorHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        if self.path != "/data":
            self.send_response(404)
            self.end_headers()
            return

        received_at = datetime.now(tz=timezone.utc).isoformat()
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            print(f"[{received_at}] JSON parse error: {e}")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Bad JSON")
            return

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

        if isinstance(data, dict) and "payload" in data:
            msg_id = data.get("messageId", "?")
            device = data.get("deviceId", "unknown")
            print(f"\n--- message {msg_id} from {device} at {received_at} ---")
            process_payload(data["payload"], received_at)
        elif isinstance(data, list):
            process_payload(data, received_at)
        else:
            print(f"[{received_at}] Raw: {json.dumps(data, indent=2)}")
            flat = flatten_values(data) if isinstance(data, dict) else {"raw": str(data)}
            row = {"received_at": received_at, "sensor_time": received_at, "sensor": "raw", **flat}
            write_csv_row(row)


def main():
    host, port = "0.0.0.0", 8000
    server = HTTPServer((host, port), SensorHandler)
    print(f"Sensor server running on {host}:{port}")
    print(f"Saving data to: {CSV_FILE}")
    print("POST sensor data to http://<your-ip>:8000/data")
    print("Press Ctrl+C to stop.\n")
    print(f"{'Timestamp':<35} {'Sensor':<22} {'Hz':>6}  Values")
    print("-" * 90)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
