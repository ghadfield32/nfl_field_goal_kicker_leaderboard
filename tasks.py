# tasks.py  ── invoke ≥2.2
from invoke import task, Context  # type: ignore
from typing import List, Optional, Union

import os
import sys
import pathlib
import tempfile
import datetime as _dt
import atexit
import socket
import contextlib
import errno


BASE_ENV = pathlib.Path(__file__).parent


# Track temporary env files for cleanup
_saved_env_files: List[str] = []


def _parse_port(port: Union[str, int, None]) -> Optional[int]:
    """
    Parse and validate a port number.
    
    Args:
        port: Port number as string or int, or None
        
    Returns:
        Validated port number as int, or None if input was None
        
    Raises:
        ValueError: If port is invalid or out of range
    """
    if port is None:
        return None
        
    try:
        port_int = int(port)
        if not (0 < port_int < 65536):
            raise ValueError(f"Port {port_int} out of valid range (1-65535)")
        return port_int
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid port value: {port}") from e


def _first_free_port(start: int = 5200) -> int:
    """Return the first TCP port >= *start* that is unused on localhost."""
    print(f"DEBUG: Searching for free port starting at {start}")  # Debug
    import socket
    import contextlib
    for port in range(start, 65535):
        with contextlib.closing(socket.socket()) as s:
            if s.connect_ex(("127.0.0.1", port)):
                print(f"DEBUG: Found free port {port}")  # Debug
                return port
    raise RuntimeError("No free port found")


def _free_port(start=5200) -> int:
    """Find a free port by letting the OS assign one."""
    print(f"DEBUG: Finding free port starting at {start}")  # Debug
    import socket
    import contextlib
    with contextlib.closing(
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
        print(f"DEBUG: Found free port {port}")  # Debug
        return port


def _port_free(host: str, port: int, timeout: float = 0.1) -> bool:
    """
    Return True iff *host:port* is NOT in use.

    Uses a non-blocking TCP connect – works on Linux, macOS, Windows,
    inside or outside WSL – and does **not** rely on lsof / netstat.
    """
    print(f"DEBUG: Checking if port {port} is free on {host}")  # Debug
    try:
        with contextlib.closing(
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ) as s:
            s.settimeout(timeout)
            s.connect((host, port))
            print(f"DEBUG: Port {port} is in use")  # Debug
            return False      # connection succeeded ⇒ something listening
    except (OSError, socket.timeout):
        print(f"DEBUG: Port {port} is free")  # Debug
        return True           # connection failed ⇒ port is free


def _find_port(preferred: int, start: int = 5200) -> int:
    """
    Try to use preferred port, fall back to finding first available port.
    
    Args:
        preferred: The preferred port number to try first
        start: Where to start searching if preferred port is taken
        
    Returns:
        An available port number
    """
    print(f"DEBUG: Trying preferred port {preferred}")  # Debug
    if _port_free("127.0.0.1", preferred):
        return preferred
    return _first_free_port(start)


def _write_envfile(name: str, 
                   ports: Optional[dict[str, int]] = None) -> pathlib.Path:
    """
    Create a throw-away .env file for the current `invoke up` run.
    
    Docker-compose will use this to see the chosen host-ports. We include all
    services we know about; anything unset falls back to .env.template defaults.
    """
    env_lines = [f"ENV_NAME={name}"]
    mapping = {
        "jupyter": "HOST_JUPYTER_PORT",
        "tensorboard": "HOST_TENSORBOARD_PORT",
        "explainer": "HOST_EXPLAINER_PORT",
        "streamlit": "HOST_STREAMLIT_PORT",
        "mlflow": "HOST_MLFLOW_PORT",      # NEW
    }
    for svc, var in mapping.items():
        if ports and svc in ports:
            env_lines.append(f"{var}={ports[svc]}")
    env_lines.append(f"# generated {_dt.datetime.now().isoformat()}")
    tmp = tempfile.NamedTemporaryFile(
        "w", 
        delete=False, 
        prefix=".env.",
        dir=BASE_ENV
    )
    tmp.write("\n".join(env_lines))
    tmp.close()
    _saved_env_files.append(tmp.name)
    return pathlib.Path(tmp.name)


# Register cleanup function
def _cleanup_env_files() -> None:
    """Remove all temporary env files."""
    for path in _saved_env_files:
        try:
            os.remove(path)
        except OSError:
            pass


atexit.register(_cleanup_env_files)


def _compose(
    c: Context,
    cmd: str,
    name: str,
    rebuild: bool = False,
    force_pty: bool = False,
    ports: Optional[dict[str, int]] = None,
) -> None:
    """
    Wrapper around `docker compose` that also sanity-checks host ports.
    """
    # ---------- NEW pre-flight check --------------------------------------
    if ports:
        for svc, port in ports.items():
            if port is None:
                continue
            if not _port_free("127.0.0.1", int(port)):
                print(f"❌  Host port {port} already bound – "
                      f"{svc} cannot start. Choose another port (invoke up "
                      f"--{svc}-port XXXXX) or free it first.")
                sys.exit(1)

    env = {**os.environ, "ENV_NAME": name, "COMPOSE_PROJECT_NAME": name}
    
    # Add port overrides if provided
    if ports:
        port_mapping = {
            "jupyter": "HOST_JUPYTER_PORT",
            "tensorboard": "HOST_TENSORBOARD_PORT", 
            "explainer": "HOST_EXPLAINER_PORT",
            "streamlit": "HOST_STREAMLIT_PORT",
        }
        for service, port in ports.items():
            if service in port_mapping:
                env[port_mapping[service]] = str(port)
    
    use_pty = force_pty or (os.name != "nt" and sys.stdin.isatty())

    if not use_pty and not getattr(_compose, "_warned", False):
        print("ℹ️  PTY not supported – running without TTY.")
        _compose._warned = True  # type: ignore[attr-defined]

    if rebuild:
        full_cmd = f"docker compose -p {name} {cmd} --build"
    else:
        full_cmd = f"docker compose -p {name} {cmd}"
    c.run(full_cmd, env=env, pty=use_pty)


@task(
    help={
        "name": "Project/venv name (defaults to folder name)",
        "use_pty": "Force PTY even on non-POSIX hosts",
        "jupyter_port": "Jupyter Lab port (default: 8890)",
        "tensorboard_port": "TensorBoard port (default: auto-assigned)",
        "explainer_port": "Explainer Dashboard port (default: auto-assigned)", 
        "streamlit_port": "Streamlit port (default: auto-assigned)",
        "mlflow_port": "MLflow UI port (default: 5000, auto-assigns if busy)",
    }
)
def up(
    c,
    name: Optional[str] = None,
    rebuild: bool = False,
    detach: bool = True,
    use_pty: bool = False,
    jupyter_port: Union[str, int, None] = None,
    tensorboard_port: Union[str, int, None] = None,
    explainer_port: Union[str, int, None] = None,
    streamlit_port: Union[str, int, None] = None,
    mlflow_port: Union[str, int, None] = None,
) -> None:
    """Build (optionally --rebuild) & start the container with custom ports."""
    name = name or BASE_ENV.name

    # ---------- Parse and validate all ports -----------------
    try:
        jupyter_port = _parse_port(jupyter_port)
        tensorboard_port = _parse_port(tensorboard_port)
        explainer_port = _parse_port(explainer_port)
        streamlit_port = _parse_port(streamlit_port)
        mlflow_port = _parse_port(mlflow_port)
    except ValueError as e:
        print(f"❌ Port validation failed: {e}")
        sys.exit(1)

    # ---------- build dynamic port map -----------------
    ports = {}
    if jupyter_port is not None:
        ports["jupyter"] = jupyter_port
    if tensorboard_port is not None:
        ports["tensorboard"] = tensorboard_port
    if explainer_port is not None:
        ports["explainer"] = explainer_port
    if streamlit_port is not None:
        ports["streamlit"] = streamlit_port

    # ---------- Explainer auto-assign (NEW) ------------
    print("DEBUG: Starting explainer port assignment")  # Debug
    try:
        # Try to use the explainer's version first
        from src.mlops.explainer import _first_free_port  # type: ignore
        print("DEBUG: Successfully imported _first_free_port from explainer")  # Debug
    except ModuleNotFoundError:
        print("DEBUG: Failed to import _first_free_port, using local implementation")  # Debug
        # We'll use our local _first_free_port implementation
        pass

    if explainer_port is None:
        print("DEBUG: No explainer port specified, finding one")  # Debug
        explainer_port = _find_port(8050, 5200)
    elif not _port_free("127.0.0.1", explainer_port):
        print(f"DEBUG: Specified explainer port {explainer_port} is in use")  # Debug
        sys.exit(1)
    ports["explainer"] = explainer_port
    print(f"🔌 Explainer host-port → {explainer_port}")

    # ----- MLflow auto-assign (default 5000) -----------
    print("DEBUG: Starting MLflow port assignment")  # Debug
    if mlflow_port is None:
        print("DEBUG: No MLflow port specified, finding one")  # Debug
        mlflow_port = _find_port(5000, 5200)
    elif not _port_free("127.0.0.1", mlflow_port):
        print(f"DEBUG: Specified MLflow port {mlflow_port} is in use")  # Debug
        sys.exit(1)
    ports["mlflow"] = mlflow_port
    print(f"🔌 MLflow host-port → {mlflow_port}")

    # Generate environment file
    env_path = _write_envfile(name, ports)
    compose_cmd = "up -d" if detach else "up"

    _compose(
        c,
        f"--env-file {env_path} {compose_cmd}",
        name,
        rebuild=rebuild,
        force_pty=use_pty,
        ports=ports,
    )


@task(
    help={
        "name": "Project/venv name (defaults to folder name)",
    }
)
def stop(c, name: Optional[str] = None) -> None:
    """Stop and remove dev container (keeps volumes)."""
    name = name or BASE_ENV.name
    cmd = f"docker compose -p {name} down"
    try:
        c.run(cmd)
        print(f"\n🛑 Stopped and removed project '{name}'")
    except Exception:
        print(f"❌ No running containers found for project '{name}'")


@task
def shell(c, name: str | None = None) -> None:
    """Open an interactive shell inside the running container."""
    name = name or BASE_ENV.name
    cmd = f"docker compose -p {name} ps -q datascience"
    cid = c.run(cmd, hide=True).stdout.strip()
    c.run(f"docker exec -it {cid} bash", env={"ENV_NAME": name}, pty=False)


@task
def clean(c) -> None:
    """Prune stopped containers + dangling images."""
    c.run("docker system prune -f")


@task
def ports(c, name: str | None = None) -> None:
    """Show current port mappings for the named project."""
    name = name or BASE_ENV.name
    cmd = f"docker compose -p {name} ps --format table"
    try:
        c.run(cmd, hide=False)
        print(f"\n📊 Port mappings for project '{name}':")
        print("=" * 50)
    except Exception:
        print(f"❌ No running containers found for project '{name}'")
        print("\n💡 Usage examples:")
        print("  invoke up --name myproject --jupyter-port 8891")
        print("  invoke up --name myproject --jupyter-port 8892 \\")
        print("    --tensorboard-port 6009")


# --- utilities ---------------------------------------------------------------
def _norm(path: str | pathlib.Path) -> str:
    """Return a lower-case, forward-slash, no-trailing-slash version of *path*."""
    p = str(path).replace("\\", "/").rstrip("/").lower()
    return p

def _docker_projects_from_this_repo() -> set[str]:
    """
    Discover every Compose *project name* whose working_dir label ends with
    the current repo path.

    Works across Windows ↔ WSL ↔ macOS because we do suffix-match on a
    normalised path.
    """
    here_tail = _norm(pathlib.Path(__file__).parent.resolve())
    cmd = (
        "docker container ls -a "
        "--format '{{.Label \"com.docker.compose.project\"}} "
        "{{.Label \"com.docker.compose.project.working_dir\"}}' "
        "--filter label=com.docker.compose.project"
    )
    projects: set[str] = set()
    for line in os.popen(cmd).read().strip().splitlines():
        try:
            proj, wd = line.split(maxsplit=1)
        except ValueError:
            continue
        if _norm(wd).endswith(here_tail):
            projects.add(proj)
    return projects

# --- task --------------------------------------------------------------------
@task(
    help={
        "name": "Project name (defaults to folder). Ignored with --all.",
        "all":  "Remove *all* projects launched from this repo.",
        "rmi":  "Image-removal policy: all | local | none (default: local).",
    }
)
def down(c, name: str | None = None, all: bool = False, rmi: str = "local"):
    """
    Stop containers **and** fully delete every artefact so next `invoke up`
    starts from a clean slate.

    Examples
    --------
    invoke down                  # nuke current-folder project
    invoke down --name ml_project --rmi all   # wipe everything for ml_project
    invoke down --all            # tear down every project from this repo
    """
    if rmi not in {"all", "local", "none"}:
        raise ValueError("--rmi must be all | local | none")

    targets = _docker_projects_from_this_repo() if all else {name or BASE_ENV.name}
    flags = "-v --remove-orphans"
    if rmi != "none":
        flags += f" --rmi {rmi}"

    for proj in targets:
        try:
            c.run(f"docker compose -p {proj} down {flags}")
            print(f"🗑️  Removed project '{proj}'")
        except Exception:
            print(f"⚠️  Nothing to remove for '{proj}'")


@task(
    help={
        "yaml": "Path to dashboard.yaml file",
        "port": "Port to serve on (default: 8150)",
        "host": "Host to bind to (default: 0.0.0.0)",
    }
)
def dashboard(c, yaml: str, port: int = 8150, host: str = "0.0.0.0") -> None:
    """
    Serve a saved ExplainerDashboard from a YAML configuration file.
    
    This task allows you to re-serve dashboards that were previously saved
    with build_and_log_dashboard(save_yaml=True).
    
    Examples:
        invoke dashboard --yaml dashboard.yaml
        invoke dashboard --yaml dashboard.yaml --port 8200
    """
    import sys
    from pathlib import Path
    from src.mlops.explainer import load_dashboard_yaml
    
    yaml_path = Path(yaml)
    if not yaml_path.exists():
        print(f"❌ Dashboard YAML file not found: {yaml_path}")
        sys.exit(1)
    
    # Check if port is available
    if not _port_free(host, port):
        print(f"❌ Port {port} is already in use on {host}")
        sys.exit(1)
    
    try:
        print(f"🔄 Loading dashboard from {yaml_path}")
        dashboard_obj = load_dashboard_yaml(yaml_path)
        
        print(f"🌐 Serving ExplainerDashboard on {host}:{port}")
        dashboard_obj.run(port=port, host=host, use_waitress=True, open_browser=False)
        
    except Exception as e:
        print(f"❌ Failed to load or serve dashboard: {e}")
        sys.exit(1)

