"""
BMW DriveIQ — AI Report API Endpoint
Calls Groq / LLaMA 3.3 70B to generate a personalised coaching report.
"""

from http.server import BaseHTTPRequestHandler
import json
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "bmw_driveiq"))

from utils.coach import generate_report


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length > 0 else {}

            stats           = body.get("stats", {})
            predicted_label = int(body.get("predicted_label", 0))
            confidence      = float(body.get("confidence", 0.9))

            report = generate_report(stats, predicted_label, confidence, verbose=False)

            payload = json.dumps({"report": report}).encode()
            self._respond(200, payload)

        except Exception as e:
            self._respond(500, json.dumps({"error": str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _respond(self, code, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass
