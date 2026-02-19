import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
import uuid


def _clean_noncritical_output(output: str) -> str:
    lines = str(output or "").splitlines()
    cleaned: list[str] = []
    skip_tokens = (
        "Matplotlib is building the font cache",
        "is not a writable directory",
        "Matplotlib created a temporary cache directory",
        "Fontconfig error: No writable cache directories",
    )

    for line in lines:
        if any(token in line for token in skip_tokens):
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _validate_code_shape(tree: ast.Module) -> tuple[bool, str | None]:
    body = list(tree.body or [])
    if not body:
        return False, "Generated code is empty"

    for idx, node in enumerate(body):
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
            if idx == 0:
                continue
            return False, "Generated code contains stray string literal block (likely commentary/docstring artifact)"

    has_executable_action = False
    for node in body:
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.While, ast.If, ast.With, ast.Try)):
            has_executable_action = True
            break
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Call):
            has_executable_action = True
            break

    if not has_executable_action:
        return False, "Generated code does not contain executable computation statements"

    return True, None


def _coerce_result_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_result_value(item) for item in value[:200]]
    if isinstance(value, dict):
        coerced = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= 200:
                break
            coerced[str(key)] = _coerce_result_value(item)
        return coerced
    return repr(value)


def _execute_code_isolated(
    code: str,
    save_plots: bool,
    plots_dir: str,
    mpl_cache_dir: str,
    xdg_cache_dir: str,
) -> dict:
    result = {
        "success": False,
        "output": "",
        "result": None,
        "plots": [],
        "error": None,
    }

    out = io.StringIO()
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache_dir)

    try:
        unsafe = {"os", "sys", "subprocess", "shutil"}

        def safe_import(name, *args, **kwargs):
            root = name.split(".")[0]
            if root in unsafe:
                raise ImportError(f"Importing '{root}' blocked")
            return __import__(name, *args, **kwargs)

        env = {"__name__": "__main__"}
        try:
            import builtins

            safe_builtins = builtins.__dict__.copy()
            for blocked in ("open", "exec", "eval", "input"):
                safe_builtins.pop(blocked, None)
            safe_builtins["__import__"] = safe_import
            env["__builtins__"] = safe_builtins
        except Exception:
            pass

        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                tree = ast.parse(code, mode="exec")
                is_valid_shape, shape_error = _validate_code_shape(tree)
                if not is_valid_shape:
                    result["error"] = f"Code shape validation failed: {shape_error}"
                    raise ValueError(result["error"])

                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    result_var = "__result__"
                    assign = ast.Assign(targets=[ast.Name(id=result_var, ctx=ast.Store())], value=tree.body[-1].value)
                    tree.body[-1] = assign
                    ast.fix_missing_locations(tree)
                else:
                    result_var = None

                # Force non-interactive plotting even if generated code calls plt.show().
                try:
                    import matplotlib

                    matplotlib.use("Agg", force=True)
                    import matplotlib.pyplot as plt

                    plt.show = lambda *args, **kwargs: None
                except Exception:
                    pass

                code_obj = compile(tree, "<string>", "exec")
                exec(code_obj, env)

                if result_var and result_var in env:
                    result["result"] = _coerce_result_value(env[result_var])
            except Exception:
                result["error"] = traceback.format_exc()

        if save_plots and result["error"] is None:
            try:
                import matplotlib.pyplot as plt

                for num in plt.get_fignums():
                    fig = plt.figure(num)
                    path = Path(plots_dir) / f"plot_{num}.png"
                    fig.savefig(path, dpi=100, bbox_inches="tight")
                    result["plots"].append(str(path.resolve()))
                plt.close("all")
            except Exception:
                pass

        result["success"] = result["error"] is None
        result["output"] = _clean_noncritical_output(out.getvalue())
    except Exception:
        result["error"] = traceback.format_exc()

    return result


def _worker_entry(args: list[str]) -> int:
    if len(args) != 6:
        return 2

    code_path = Path(args[0])
    result_path = Path(args[1])
    save_plots = args[2] == "1"
    plots_dir = args[3]
    mpl_cache_dir = args[4]
    xdg_cache_dir = args[5]

    try:
        code = code_path.read_text(encoding="utf-8")
        result = _execute_code_isolated(
            code=code,
            save_plots=save_plots,
            plots_dir=plots_dir,
            mpl_cache_dir=mpl_cache_dir,
            xdg_cache_dir=xdg_cache_dir,
        )
    except Exception:
        result = {
            "success": False,
            "output": "",
            "result": None,
            "plots": [],
            "error": traceback.format_exc(),
        }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")
    return 0


def python_exec_tool(code: str, install_packages: list = None, timeout: int = 30, save_plots: bool = True) -> dict:
    """Run Python code in a sandboxed subprocess with a hard timeout."""
    if install_packages is None:
        install_packages = []

    result = {
        "success": False,
        "output": "",
        "result": None,
        "plots": [],
        "error": None,
        "installed_packages": [],
    }

    safe_packages = {
        "numpy", "pandas", "matplotlib", "seaborn", "plotly", "scipy", "sympy",
        "scikit-learn", "requests", "beautifulsoup4", "pillow", "statsmodels", "mpmath"
    }
    install_packages = [pkg for pkg in install_packages if pkg.lower() in safe_packages]

    for pkg in install_packages:
        try:
            proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True, timeout=60)
            if proc.returncode == 0:
                result["installed_packages"].append(pkg)
        except Exception:
            pass

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    plots_dir = Path("temp") / "plots" / run_id
    plots_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = Path("temp") / "matplotlib_cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache = Path("temp") / "xdg_cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    worker_dir = Path("temp") / "code_exec" / run_id
    worker_dir.mkdir(parents=True, exist_ok=True)
    code_path = worker_dir / "input_code.py"
    result_path = worker_dir / "result.json"
    code_path.write_text(code or "", encoding="utf-8")

    effective_timeout = max(1, int(timeout))
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(code_path.resolve()),
        str(result_path.resolve()),
        "1" if save_plots else "0",
        str(plots_dir.resolve()),
        str(matplotlib_cache.resolve()),
        str(xdg_cache.resolve()),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=effective_timeout + 1,
        )
    except subprocess.TimeoutExpired:
        result["error"] = f"Execution timed out after {effective_timeout} seconds"
        result["success"] = False
        return result
    except Exception:
        result["error"] = traceback.format_exc()
        result["success"] = False
        return result

    if result_path.exists():
        try:
            child_result = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(child_result, dict):
                result.update(child_result)
        except Exception:
            result["error"] = f"Failed to parse execution result: {traceback.format_exc()}"
            result["success"] = False
    else:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        details = stderr or stdout or f"worker exited with code {proc.returncode}"
        result["error"] = f"Execution subprocess produced no result file: {details}"
        result["success"] = False

    if proc.returncode != 0 and not result.get("error"):
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        result["error"] = stderr or stdout or f"Execution subprocess failed with exit code {proc.returncode}"
        result["success"] = False

    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_entry(sys.argv[2:]))
