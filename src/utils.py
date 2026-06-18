import os
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("Omega.Utils")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

# ── Project root resolution ────────────────────────────────────────────────────
# utils.py lives at src/utils.py → parent is src/ → parent.parent is project root
_PROJECT_ROOT = Path(__file__).parent.parent
_OUTPUT_DIR   = _PROJECT_ROOT / "output"


# ── Path helpers ───────────────────────────────────────────────────────────────

def get_output_path(filename: str) -> str:
    """
    Return the absolute path for a file in the output/ directory.
    Creates the output/ directory if it does not exist.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(_OUTPUT_DIR / filename)


def get_config_path(filename: str) -> str:
    """
    Return the absolute path for a file in the config/ directory.
    """
    config_dir = _PROJECT_ROOT / "config"
    return str(config_dir / filename)


# ── File I/O ───────────────────────────────────────────────────────────────────

def load_json_file(path: Any) -> Dict[str, Any]:
    """
    Load and parse a JSON file from disk.
    Accepts a str path or a Path object.
    Raises json.JSONDecodeError if the file is not valid JSON.
    Raises FileNotFoundError if the file does not exist.
    """
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_output(filename: str) -> Optional[Dict[str, Any]]:
    """
    Safely load a JSON file from the output/ directory.
    Returns None (never raises) if the file is missing or malformed.
    """
    path = get_output_path(filename)
    if not os.path.exists(path):
        return None
    try:
        return load_json_file(path)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load output file '%s': %s", filename, exc)
        return None


def load_md_output(filename: str) -> Optional[str]:
    """
    Safely load a markdown file from the output/ directory.
    Returns None if the file is missing or unreadable.
    """
    path = get_output_path(filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        logger.warning("Could not load markdown file '%s': %s", filename, exc)
        return None


def clear_output_dir() -> None:
    """
    Delete all files in the output/ directory.
    Called by app.py at the start of each new query to avoid
    stale results from a previous run being rendered.
    """
    if not _OUTPUT_DIR.exists():
        return
    for file in _OUTPUT_DIR.iterdir():
        if file.is_file():
            try:
                file.unlink()
            except OSError as exc:
                logger.warning("Could not delete output file '%s': %s", file.name, exc)
    logger.info("Output directory cleared")


# ── Background thread runner ───────────────────────────────────────────────────
# The crew kickoff blocks for 10–60 seconds depending on query complexity.
# Running it in a background thread keeps Streamlit's event loop responsive
# so the UI can show live progress updates while the crew is running.

class CrewRunner:
    """
    Wraps a crew kickoff in a background thread.
    Exposes is_running, is_done, result, and error for app.py to poll.

    Usage in app.py:
        runner = CrewRunner(target_fn=run_omega, kwargs={...})
        runner.start()
        while runner.is_running:
            time.sleep(0.5)
            st.rerun()
        if runner.error:
            st.error(runner.error)
        else:
            result = runner.result
    """

    def __init__(self, target_fn: Callable, kwargs: Optional[Dict[str, Any]] = None):
        self._target_fn  = target_fn
        self._kwargs     = kwargs or {}
        self._thread:    Optional[threading.Thread] = None
        self._result:    Any  = None
        self._error:     Optional[str] = None
        self._is_done:   bool = False
        self._lock       = threading.Lock()

    def _run(self) -> None:
        try:
            result = self._target_fn(**self._kwargs)
            with self._lock:
                self._result  = result
                self._is_done = True
        except Exception as exc:
            logger.exception(f"CrewRunner thread exception: {exc}")
            with self._lock:
                self._error   = str(exc)
                self._is_done = True

    def start(self) -> None:
        """Start the background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("CrewRunner thread started")

    @property
    def is_running(self) -> bool:
        """True while the crew is still executing."""
        with self._lock:
            return not self._is_done

    @property
    def is_done(self) -> bool:
        """True once the crew has finished (success or failure)."""
        with self._lock:
            return self._is_done

    @property
    def result(self) -> Any:
        """The return value of the crew kickoff, or None if not yet done."""
        with self._lock:
            return self._result

    @property
    def error(self) -> Optional[str]:
        """Error message string if the crew raised an exception, else None."""
        with self._lock:
            return self._error


# ── Progress tracking ──────────────────────────────────────────────────────────
# CrewAI fires task_callback after each task completes.
# OmegaProgressTracker maps task names to human-readable labels and
# updates st.session_state so app.py can render a live progress bar.

# Maps the task function name (as CrewAI resolves it) to a display label
TASK_LABELS: Dict[str, str] = {
    "run_eda":          "Analysing dataset structure",
    "run_query":        "Running your query",
    "render_chart":     "Building chart",
    "generate_insight": "Composing insights",
}

TASK_ORDER = ["run_eda", "run_query", "render_chart", "generate_insight"]


class OmegaProgressTracker:
    """
    Stateful progress tracker that maps CrewAI task callbacks to
    a list of completed steps, stored in a dict for app.py to read.

    Usage in app.py:
        tracker = OmegaProgressTracker()
        runner = CrewRunner(
            target_fn=run_omega,
            kwargs={"task_callback": tracker.on_task_complete, ...}
        )
    """

    def __init__(self) -> None:
        self._completed: list[str] = []
        self._lock = threading.Lock()

    def on_task_complete(self, task_output: Any) -> None:
        """
        Called by CrewAI after each task finishes.
        Resolves the task name and appends to the completed list.
        """
        task_name = self._resolve_task_name(task_output)
        label     = TASK_LABELS.get(task_name, task_name.replace("_", " ").title())
        with self._lock:
            if task_name not in self._completed:
                self._completed.append(task_name)
        logger.info(f"Task complete — {task_name} ({label})")

    def _resolve_task_name(self, task_output: Any) -> str:
        """Extract the task key from a CrewAI task output object."""
        raw = (
            getattr(task_output, "name", "")
            or getattr(task_output, "description", "")
            or ""
        )
        normalised = str(raw).lower().replace(" ", "_")
        for key in TASK_ORDER:
            if key in normalised:
                return key
        return normalised

    @property
    def completed_tasks(self) -> list[str]:
        with self._lock:
            return list(self._completed)

    @property
    def completed_count(self) -> int:
        with self._lock:
            return len(self._completed)

    @property
    def total_tasks(self) -> int:
        return len(TASK_ORDER)

    @property
    def progress_fraction(self) -> float:
        """Float between 0.0 and 1.0 for st.progress()."""
        return min(1.0, self.completed_count / max(1, self.total_tasks))

    def get_status_lines(self) -> list[Dict[str, Any]]:
        """
        Return a list of dicts for app.py to render as a step tracker.
        Each dict has: task_name, label, status (done | running | pending)
        """
        completed = self.completed_tasks
        lines = []
        found_running = False
        for task in TASK_ORDER:
            if task in completed:
                status = "done"
            elif not found_running:
                status = "running"
                found_running = True
            else:
                status = "pending"
            lines.append({
                "task_name": task,
                "label":     TASK_LABELS.get(task, task),
                "status":    status,
            })
        return lines


# ── Output polling helper ──────────────────────────────────────────────────────

def wait_for_output(filename: str,
                    attempts: int = 15,
                    delay_seconds: float = 0.3) -> Optional[Dict[str, Any]]:
    """
    Poll for an output file written by an agent, with retries.
    Used by app.py after the crew finishes to safely load results.

    Returns the parsed JSON dict, or None if the file never appeared.
    """
    path = Path(get_output_path(filename))
    for attempt in range(attempts):
        if path.exists():
            try:
                return load_json_file(path)
            except json.JSONDecodeError:
                logger.warning(
                    "File '%s' exists but is not yet valid JSON — retry %d/%d",
                    filename, attempt + 1, attempts
                )
        time.sleep(delay_seconds)
    logger.warning("File '%s' not available after %d attempts", filename, attempts)
    return None