from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from aiohttp import ClientSession, WSMsgType, web


ROOT = Path(__file__).resolve().parent
UPSTREAM_HTTP = "http://127.0.0.1:3001"
UPSTREAM_WS = "ws://127.0.0.1:3001"
PUBLIC_PORT = 3000


def start_streamlit() -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.address=127.0.0.1",
        "--server.port=3001",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--browser.gatherUsageStats=false",
    ]
    return subprocess.Popen(command, cwd=ROOT, env=os.environ.copy())


def wait_for_streamlit(process: subprocess.Popen, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"{UPSTREAM_HTTP}/_stcore/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Streamlit exited early with status {process.returncode}")
        try:
            with urlopen(health_url, timeout=1) as response:
                if 200 <= response.status < 300:
                    print("FraudShield Streamlit process is ready on the internal port.", flush=True)
                    return
        except (URLError, TimeoutError, OSError):
            time.sleep(0.35)
    raise TimeoutError("Streamlit did not become ready before the preview startup deadline.")


async def serve_manifest(_: web.Request) -> web.FileResponse:
    manifest = ROOT / "public" / "manus-routes.json"
    return web.FileResponse(manifest, headers={"Cache-Control": "no-cache", "Content-Type": "application/json; charset=utf-8"})


async def proxy_request(request: web.Request) -> web.StreamResponse:
    path_qs = str(request.rel_url)
    if request.headers.get("Upgrade", "").lower() == "websocket":
        protocols = [protocol.strip() for protocol in request.headers.get("Sec-WebSocket-Protocol", "").split(",") if protocol.strip()]
        client_socket = web.WebSocketResponse(protocols=protocols, heartbeat=25, autoping=True)
        await client_socket.prepare(request)
        upstream_url = f"{UPSTREAM_WS}{path_qs}"
        origin = request.headers.get("Origin", "http://127.0.0.1:3000")
        try:
            async with ClientSession() as session:
                async with session.ws_connect(upstream_url, headers={"Origin": origin}, protocols=protocols, heartbeat=25, autoping=True) as upstream_socket:
                    async def client_to_upstream() -> None:
                        async for message in client_socket:
                            if message.type == WSMsgType.TEXT:
                                await upstream_socket.send_str(message.data)
                            elif message.type == WSMsgType.BINARY:
                                await upstream_socket.send_bytes(message.data)
                            elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                                break

                    async def upstream_to_client() -> None:
                        async for message in upstream_socket:
                            if message.type == WSMsgType.TEXT:
                                await client_socket.send_str(message.data)
                            elif message.type == WSMsgType.BINARY:
                                await client_socket.send_bytes(message.data)
                            elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                                break

                    await asyncio.gather(client_to_upstream(), upstream_to_client())
        except Exception as exc:  # The preview proxy should close cleanly if Streamlit restarts.
            print(f"WebSocket proxy closed: {exc}", flush=True)
        finally:
            if not client_socket.closed:
                await client_socket.close()
        return client_socket

    target = f"{UPSTREAM_HTTP}{path_qs}"
    hop_by_hop = {"host", "connection", "upgrade", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding"}
    headers = {name: value for name, value in request.headers.items() if name.lower() not in hop_by_hop}
    body = await request.read()
    timeout = None
    async with ClientSession(timeout=timeout) as session:
        async with session.request(request.method, target, headers=headers, data=body, allow_redirects=False) as response:
            response_body = await response.read()
            response_headers = {name: value for name, value in response.headers.items() if name.lower() not in hop_by_hop and name.lower() not in {"content-length", "content-encoding"}}
            return web.Response(body=response_body, status=response.status, headers=response_headers)


async def run_proxy(process: subprocess.Popen) -> None:
    application = web.Application(client_max_size=16 * 1024 * 1024)
    application.router.add_get("/manus-routes.json", serve_manifest)
    application.router.add_route("*", "/{path:.*}", proxy_request)
    runner = web.AppRunner(application, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=PUBLIC_PORT)
    await site.start()
    print(f"FraudShield Preview proxy listening on 0.0.0.0:{PUBLIC_PORT}.", flush=True)
    try:
        while process.poll() is None:
            await asyncio.sleep(1)
    finally:
        await runner.cleanup()


def main() -> None:
    process = start_streamlit()
    try:
        wait_for_streamlit(process)
        asyncio.run(run_proxy(process))
    except KeyboardInterrupt:
        pass
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
