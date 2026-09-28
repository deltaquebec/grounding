"""shared utilities for grounding audits"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PREFIXES = ("fig", "tab", "log", "data", "model", "meta")
EXCLUDED_OPTIONS = {"--root", "--name", "--label"}


def find_root(script_path: Path) -> Path:
    """project root"""
    here = Path(script_path).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return here


def sanitize(text: str) -> str:
    """lowercase; characters outside letters, digits, hyphen, period mapped to hyphen"""
    return re.sub(r"[^a-z0-9.-]", "-", text.lower())


def add_run_options(parser: argparse.ArgumentParser) -> None:
    """--root, --name, --label options"""
    parser.add_argument("--root", type=Path, default=None, help="project root override")
    parser.add_argument("--name", type=str, default=None, help="outputs subfolder, default script stem")
    parser.add_argument("--label", type=str, default=None, help="run label, replaces the derived one")


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def derive_label(parser: argparse.ArgumentParser, argv: Sequence[str], stem: str,
                 path_options: set[str] = frozenset()) -> str:
    """label from arguments in order passed"""
    option_map = {}
    positional_actions = []
    for action in parser._actions:
        for opt in action.option_strings:
            option_map[opt] = action
        if not action.option_strings and action.dest != "help":
            positional_actions.append(action)
    parts: list[str] = []
    override = None
    positional_index = 0
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("-") and len(token) > 1:
            key, has_eq, value = token.partition("=")
            action = option_map.get(key)
            if action is None:
                i += 1
                continue
            consumes = action.nargs != 0
            if not has_eq and consumes:
                j = i + 1
                values = []
                while j < len(argv) and not (argv[j].startswith("-") and len(argv[j]) > 1 and not _is_number(argv[j])):
                    values.append(argv[j])
                    j += 1
                    if action.nargs in (None, 1):
                        break
                value = "-".join(values)
                i = j
            else:
                i += 1
            if key == "--label":
                override = value
                continue
            if key in EXCLUDED_OPTIONS:
                continue
            name = key.lstrip("-")
            if not consumes:
                parts.append(name)
                continue
            if key in path_options or action.type is Path:
                value = Path(value).stem
            parts.append(f"{name}-{value}")
        else:
            action = positional_actions[positional_index] if positional_index < len(positional_actions) else None
            positional_index += 1
            value = Path(token).stem if action is not None and action.type is Path else token
            parts.append(value)
            i += 1
    if override is not None:
        return sanitize(override)
    if not parts:
        return f"output_{sanitize(stem)}"
    return sanitize("-".join(parts))


def git_info(root: Path) -> dict[str, Any]:
    """when root is under git, else empty"""
    if not (root / ".git").exists():
        return {}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                                capture_output=True, text=True).stdout.strip()
        return {"git_commit": commit, "git_dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {}


def update_latest(base: Path, run_dir: Path) -> None:
    """point base/latest at run_dir; atomic replace where allows"""
    latest = base / "latest"
    if os.name == "nt":
        if latest.is_symlink():
            latest.unlink()
        elif latest.exists():
            os.rmdir(latest)
        try:
            os.symlink(run_dir, latest, target_is_directory=True)
        except OSError:
            subprocess.run(["cmd", "/c", "mklink", "/J", str(latest), str(run_dir)],
                           check=True, capture_output=True)
        return
    temporary = base / f".latest-{os.getpid()}-{time.time_ns()}"
    os.symlink(run_dir.name, temporary, target_is_directory=True)
    os.replace(temporary, latest)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class RunContext:
    """folder, logger, metadata, latest link"""

    def __init__(self, script_path: Path, parser: argparse.ArgumentParser, args: argparse.Namespace,
                 argv: Sequence[str], path_options: Sequence[str] = ()):
        self.start = datetime.now()
        self.stamp = self.start.strftime("%Y-%m-%d_%H-%M-%S")
        self.script = Path(script_path).resolve()
        self.root = Path(args.root).resolve() if getattr(args, "root", None) else find_root(self.script)
        self.name = args.name if getattr(args, "name", None) else self.script.stem
        self.label = derive_label(parser, argv, self.script.stem, set(path_options))
        self.base = self.root / "outputs" / self.name
        self.base.mkdir(parents=True, exist_ok=True)
        self.dir = self._create_run_dir(self.base, f"{self.stamp}_{self.label}")
        self.tmp = self.dir / "tmp"
        self.tmp.mkdir()
        self.log = self._setup_logging()
        self.status = "running"
        self._write_meta(args, argv)
        self.log.info("run folder %s", self.dir)

    @staticmethod
    def _create_run_dir(base: Path, stem: str) -> Path:
        candidate = base / stem
        ordinal = 1
        while True:
            try:
                candidate.mkdir(exist_ok=False)
                return candidate
            except FileExistsError:
                ordinal += 1
                candidate = base / f"{stem}_{ordinal:02d}"

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        handle = logging.FileHandler(self.dir / "log_run.txt", encoding="utf-8")
        handle.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(handle)
        return logger

    def _write_meta(self, args: argparse.Namespace, argv: Sequence[str]) -> None:
        meta = {
            "script": str(self.script),
            "name": self.name,
            "label": self.label,
            "stamp": self.stamp,
            "root": str(self.root),
            "command_line": " ".join([sys.executable, str(self.script), *argv]),
            "arguments": _jsonable(vars(args)),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        meta.update(git_info(self.root))
        self.write_json("config", meta)

    def path(self, prefix: str, description: str, extension: str) -> Path:
        """file path inside run folder"""
        if prefix not in PREFIXES:
            raise ValueError(f"unknown prefix {prefix}")
        return self.dir / f"{prefix}_{sanitize(description).replace('-', '_')}.{extension}"

    def write_json(self, description: str, payload: dict[str, Any], prefix: str = "meta") -> Path:
        target = self.path(prefix, description, "json")
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2)
        return target

    def finish(self, status: str) -> None:
        """final status line, scratch removal, latest link on completion"""
        self.status = status
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)
        if status == "completed":
            try:
                update_latest(self.base, self.dir)
            except OSError as error:
                self.log.warning("latest link not updated: %s", error)
        self.log.info("status: %s", status)
        for handler in list(self.log.handlers):
            handler.flush()
            handler.close()
            self.log.removeHandler(handler)


def run_script(script_path: Path, parser: argparse.ArgumentParser, main, path_options: Sequence[str] = ()) -> None:
    """parse, open run context, call main(ctx, args), record status"""
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    ctx = RunContext(script_path, parser, args, argv, path_options)
    try:
        main(ctx, args)
    except BaseException as error:  # noqa: BLE001
        ctx.log.exception("uncaught exception")
        ctx.finish(f"failed: {type(error).__name__}: {error}")
        raise SystemExit(1) from error
    ctx.finish("completed")


@dataclass
class EvaluationTuple:
    """E = (k, t, U, P); spurious covariates for P^do, none meaning P^do = P"""
    context: str
    meaning_type: str
    threats: list[str]
    reference: str
    spurious_covariates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Tolerances:
    """None marks undeclared tolerance"""
    eps_pres: float | None = None
    eps_faith: float | None = None
    delta_comp: float | None = None
    eta: float | None = None
    tau: float | None = None
    alpha: float = 0.10
    lipschitz: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def quantile_bound(values: Sequence[float], alpha: float) -> float:
    """least w with Pr[X <= w] >= 1 - alpha on empirical distribution"""
    return float(np.quantile(np.asarray(values, dtype=float), 1.0 - alpha, method="higher"))


def robustness_verdict(eps_grid: Sequence[float], omega: Sequence[float], lipschitz: float | None) -> str:
    """G3 verdict against Lipschitz bound, or undeclared"""
    if lipschitz is None:
        return "bound undeclared; curve reported"
    holds = all(w <= lipschitz * e + 1e-12 for e, w in zip(eps_grid, omega))
    return f"holds at L={lipschitz:g}" if holds else f"fails at L={lipschitz:g}"


def tolerance_verdict(value: float, bound: float | None, name: str) -> str:
    """pass or fail of measured summary against bound"""
    if bound is None:
        return f"{name}: bound undeclared"
    return f"{name}: {'pass' if value <= bound else 'fail'} at {bound:g}"


def bootstrap_difference(on: np.ndarray, off: np.ndarray, rng: np.random.Generator,
                         n_boot: int = 1000) -> tuple[float, float]:
    """percentile 95 percent interval for mean(on) - mean(off) under independent resampling"""
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = rng.choice(on, len(on)).mean() - rng.choice(off, len(off)).mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def format_cell(value: Any, precision: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    if isinstance(value, (np.floating,)):
        return f"{float(value):.{precision}f}"
    return str(value)


def write_table(ctx: RunContext, description: str, headings: Sequence[str], rows: Sequence[Sequence[Any]],
                precision: int = 3, column_spec: str | None = None) -> tuple[Path, Path]:
    """csv and LaTeX tabular"""
    csv_path = ctx.path("tab", description, "csv")
    tex_path = ctx.path("tab", description, "tex")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(str(h) for h in headings) + "\n")
        for row in rows:
            handle.write(",".join(format_cell(v, precision) for v in row) + "\n")
    spec = column_spec or "l" * len(headings)
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & ".join(f"\\textbf{{{h}}}" for h in headings) + " \\\\", "\\midrule"]
    for row in rows:
        lines.append(" & ".join(format_cell(v, precision).replace("_", "\\_") for v in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, tex_path


LINE_STYLES = (
    {"color": "0.0", "linestyle": "-", "marker": "o"},
    {"color": "0.35", "linestyle": "--", "marker": "s"},
    {"color": "0.0", "linestyle": ":", "marker": "^"},
    {"color": "0.55", "linestyle": "-.", "marker": "D"},
    {"color": "0.2", "linestyle": (0, (5, 1, 1, 1)), "marker": "v"},
)


def line_style(index: int) -> dict[str, Any]:
    """style for index-th series"""
    style = dict(LINE_STYLES[index % len(LINE_STYLES)])
    style.update({"markerfacecolor": "white", "markeredgecolor": style["color"], "markersize": 5})
    return style


def save_figure(ctx: RunContext, fig, description: str) -> tuple[Path, Path]:
    """pdf for LaTeX and png for viewing"""
    pdf_path = ctx.path("fig", description, "pdf")
    png_path = ctx.path("fig", description, "png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    return pdf_path, png_path
