"""Loopback-only read service with a local session token and no mutation routes."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
from urllib.parse import parse_qs, unquote, urlparse

from application.research_query import ResearchQuery, MAX_RESPONSE
from infrastructure.research_store import ResearchError


def make_server(store, authorization_id, *, port=0):
    query = ResearchQuery(store, authorization_id)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log tokens, artifact IDs or local paths to public logs.

        def send(self, code, content, media_type="application/json", *, attachment=False):
            if len(content) > MAX_RESPONSE:
                code, content, media_type = 413, b'{"error":{"code":"TOO_LARGE"}}', "application/json"
            self.send_response(code)
            self.send_header("Content-Type", media_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            if attachment:
                self.send_header("Content-Disposition", 'attachment; filename="research-export"')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            expected_host = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected_host or self.headers.get("Origin") not in (None, "http://" + expected_host):
                self.send(403, b'{"error":{"code":"UNAUTHORIZED_ORIGIN"}}')
                return
            if path in {"/", "/app.js", "/style.css"}:
                name, media = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                               "/style.css": ("style.css", "text/css")}[path]
                self.send(200, (Path(__file__).with_name("ui") / name).read_bytes(), media)
                return
            if not secrets.compare_digest(self.headers.get("X-Session-Token", ""), token):
                self.send(401, b'{"error":{"code":"SESSION_REQUIRED"}}')
                return
            try:
                parameters = parse_qs(parsed.query)
                if path in {"/api/studies", "/api/runs", "/api/comparisons"}:
                    if "study_id" in parameters and parameters["study_id"] != [store.manifest("authorization-" + authorization_id)["study_id"]]:
                        raise ResearchError("UNAUTHORIZED_DATA", "requested study is outside session scope")
                    kind = {"/api/studies": "study", "/api/runs": "run", "/api/comparisons": "comparison"}[path]
                    value = query.list(kind, arm_id=parameters.get("arm_id", [None])[0], state=parameters.get("state", [None])[0],
                                       **{key: parameters.get(key, [None])[0] for key in
                                          ("model", "version", "horizon", "seed", "trainer", "predictor")},
                                       limit=int(parameters.get("limit", [50])[0]), cursor=parameters.get("cursor", [None])[0])
                elif path.startswith("/api/runs/"):
                    value = query.run(path.removeprefix("/api/runs/"))
                elif path.startswith("/api/comparisons/"):
                    value = query.comparison(path.removeprefix("/api/comparisons/"))
                elif path.startswith("/api/artifacts/"):
                    if parameters.get("manifest") == ["1"]:
                        value = query.result_manifest(path.removeprefix("/api/artifacts/"))
                        self.send(200, json.dumps(value, allow_nan=False).encode(), attachment=True)
                        return
                    exported = parameters.get("download") == ["1"]
                    content, media = query.artifact(path.removeprefix("/api/artifacts/"), export=exported)
                    self.send(200, content, media, attachment=exported)
                    return
                else:
                    self.send(404, b'{"error":{"code":"NOT_FOUND"}}')
                    return
                self.send(200, json.dumps(value, allow_nan=False).encode())
            except ResearchError as exc:
                self.send(403 if exc.code.startswith("UNAUTHORIZED") else 409, json.dumps({"error": exc.envelope()}).encode())
            except (ValueError, KeyError, OSError):
                self.send(400, b'{"error":{"code":"INVALID_OR_UNAVAILABLE_RESULT"}}')

        def do_POST(self):
            self.send(405, b'{"error":{"code":"READ_ONLY"}}')

        do_PUT = do_PATCH = do_DELETE = do_POST

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.session_token = token
    return server


def serve(store, authorization_id, port=0):
    server = make_server(store, authorization_id, port=port)
    print(f"http://127.0.0.1:{server.server_port}/#session={server.session_token}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
