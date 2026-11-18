"""Own separate Showdown server(s) while running train.py; forward training flags."""
import argparse
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def port_in_use(port):
    with socket.socket() as check:
        return check.connect_ex(("127.0.0.1", port)) == 0


def start_server(server_dir, port, logs, flags, skip_build):
    # Showdown's launcher runs its build first unless told not to; concurrent
    # builds rewrite shared dist/ files and crash servers reading them mid-write.
    command = ["node", "pokemon-showdown", "start", *(["--skip-build"] if skip_build else []),
        "--no-security", "--port", str(port)]
    with (logs/f"server-{port}.out.log").open("w") as out, (logs/f"server-{port}.err.log").open("w") as err:
        return subprocess.Popen(command, cwd=server_dir, stdout=out, stderr=err, creationflags=flags)


def wait_until_listening(server, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Showdown on port {port} exited during startup; inspect logs/server-{port}.err.log")
        if port_in_use(port):
            return
        time.sleep(1)
    raise RuntimeError(f"Showdown startup timed out on port {port}")


def stop(servers, flags):
    for server in servers:
        if server.poll() is None:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
            else:
                server.terminate()
                server.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-dir", required=True, help="Built Pokemon Showdown checkout with rlspawn plugin")
    parser.add_argument("--port", type=int, default=8002, help="First server port")
    parser.add_argument("--servers", type=int, default=1,
        help="Showdown processes on consecutive ports (training only); each runs on its own CPU core")
    parser.add_argument("--entry", choices=("train", "collect", "evaluate"), default="train")
    args, training_args = parser.parse_known_args()
    server_dir = Path(args.server_dir).resolve()
    if not (server_dir / "pokemon-showdown").is_file():
        parser.error("--server-dir must contain the pokemon-showdown Node entry point")
    if args.entry == "train" and not (server_dir/"dist/server/chat-plugins/rlspawn.js").is_file():
        parser.error("Build the bundled rlspawn plugin into Showdown before using the training launcher")
    if args.servers < 1 or (args.servers > 1 and args.entry != "train"):
        parser.error("--servers must be positive, and only training uses more than one server")
    ports = [args.port + index for index in range(args.servers)]
    busy = [port for port in ports if port_in_use(port)]
    if busy:
        parser.error(f"Port(s) already in use: {busy}")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    servers = []
    try:
        # Start servers one at a time. Only the first builds; once it listens the
        # build is complete, and the rest reuse it without rewriting shared files.
        for index, port in enumerate(ports):
            servers.append(start_server(server_dir, port, logs, flags, skip_build=index > 0))
            wait_until_listening(servers[-1], port, timeout=180 if index == 0 else 60)
        port_flags = ["--ports", *map(str, ports)] if len(ports) > 1 else ["--port", str(args.port)]
        command = [sys.executable, "-u", str(ROOT / f"{args.entry}.py"), *port_flags, *training_args]
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)
    finally:
        stop(servers, flags)


if __name__ == "__main__":
    main()
