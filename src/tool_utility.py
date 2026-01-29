try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import sys
import os
import io
import ast
import textwrap
import tempfile
import subprocess
import traceback
import contextlib
import base64
import time
import logging

logger = logging.getLogger(__name__)

_LINK_CACHE: dict[tuple[str, int], list[str]] = {}
_TEXT_CACHE: dict[str, str | None] = {}


def retrieve_links(query, max_results=5, max_retries=3, base_delay=1.0):
    cache_key = (query, max_results)
    if cache_key in _LINK_CACHE:
        return _LINK_CACHE[cache_key]

    links = []
    last_error = None
    
    for attempt in range(max_retries):
        try:
            with DDGS(timeout=20) as ddgs:
                try:
                    results = list(ddgs.text(
                        query,
                        max_results=max_results,
                        region="wt-wt",
                        safesearch="moderate",
                        timelimit="y",
                    ))
                except Exception as e:
                    last_error = e
                    results = []

                if not results:
                    try:
                        results = list(ddgs.text(
                            query,
                            max_results=max_results,
                            region="wt-wt",
                            safesearch="moderate",
                            timelimit="y",
                            backend="html",
                        ))
                    except Exception as e:
                        last_error = e
                        results = []

                for r in results:
                    url = (
                        r.get("href")
                        or r.get("url")
                        or r.get("link")
                    )
                    if url:
                        links.append(url)

            if links:
                links = list(dict.fromkeys(links))
                _LINK_CACHE[cache_key] = links
                return links
                
        except Exception as e:
            last_error = e
            
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.warning(f"DuckDuckGo search failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {last_error}")
            time.sleep(delay)
    
    logger.error(f"DuckDuckGo search failed after {max_retries} attempts for query '{query[:50]}...': {last_error}")
    _LINK_CACHE[cache_key] = []
    return []
    return links

def retrieve_text(url):
    """
    Retrieve and clean text content from a URL.
    
    Args:
        url (str): URL to retrieve text from

    Returns:
        str: Cleaned text content
    """
    if url in _TEXT_CACHE:
        return _TEXT_CACHE[url]

    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        _TEXT_CACHE[url] = text
        return text
    except Exception:
        _TEXT_CACHE[url] = None
        return None

def retrieve_text_many(urls):
    """
    Retrieve and clean text content from multiple URLs.
    
    Args:
        urls (list): List of URLs to retrieve text from
    
    Returns:
        dict: Mapping of URL to cleaned text content
    """
    results = {}
    for url in urls:
        try:
            results[url] = retrieve_text(url)
        except Exception:
            results[url] = None
    return results

def _check_safety(tree):
    """
    Check for dangerous imports or function calls in the AST.
    
    Args:
        tree (ast.AST): The AST object to check.
        
    Raises:
        ValueError: If dangerous code is detected.
    """
    unsafe_modules = {'os', 'sys', 'subprocess', 'shutil'}
    unsafe_calls = {'exec', 'eval', 'open'}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in unsafe_modules:
                    raise ValueError(f"Importing '{alias.name}' is not allowed.")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in unsafe_modules:
                raise ValueError(f"Importing from '{node.module}' is not allowed.")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in unsafe_calls:
                 raise ValueError(f"Function '{node.func.id}' is not allowed.")

def _prepare_code(code):
    """Prepare Python code for execution.
    
    Args:
        code (str): Python code to prepare
    
    Returns:
        tuple: (code object, result variable name or None)"""
    dedented = textwrap.dedent(code)
    tree = ast.parse(dedented, mode="exec")
    _check_safety(tree)
    if not tree.body:
        code_obj = compile(tree, "<string>", "exec")
        return code_obj, None
    last = tree.body[-1]
    result_var = None
    if isinstance(last, ast.Expr):
        result_var = "__python_exec_tool_result__"
        assign = ast.Assign(
            targets=[ast.Name(id=result_var, ctx=ast.Store())],
            value=last.value,
        )
        tree.body[-1] = assign
        ast.fix_missing_locations(tree)
    code_obj = compile(tree, "<string>", "exec")
    return code_obj, result_var

def python_exec_tool(code, install_packages=None, timeout=30, save_plots=True):
    """
    Execute Python code in a sandboxed environment with support for:
    - Data analysis (pandas, numpy, scipy, etc.)
    - Mathematical computations (sympy, etc.)
    - Data visualization (matplotlib, seaborn, plotly)
    - Automatic package installation
    - Plot/chart saving to temporary files
    
    Args:
        code (str): Python code to execute
        install_packages (list, optional): List of packages to install before execution
        timeout (int): Maximum execution time in seconds (default: 30)
        save_plots (bool): Whether to save matplotlib/seaborn plots to temp files
        
    Returns:
        dict: {
            'success': bool,
            'output': str,  # stdout/stderr output
            'result': Any,  # return value if code returns something
            'plots': list,  # list of paths to saved plot files
            'error': str,   # error message if failed
            'installed_packages': list  # packages that were installed
        }
    """
    if install_packages is None:
        install_packages = []
    
    safe_packages = {'numpy', 'pandas', 'matplotlib', 'seaborn', 'plotly', 'scipy', 'sympy', 'scikit-learn', 'requests', 'beautifulsoup4', 'pillow'}
    install_packages = [pkg for pkg in install_packages if pkg.lower() in safe_packages]
    
    installed_packages = []
    pip_output = ""
    for pkg in install_packages:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg],
                capture_output=True,
                text=True,
            )
            pip_output += proc.stdout + proc.stderr
            if proc.returncode == 0:
                installed_packages.append(pkg)
        except Exception as e:
            pip_output += f"Failed to install {pkg}: {e}\n"
    temp_dir = tempfile.mkdtemp(prefix="python_exec_tool_")
    # Run in-process so the tool can be imported without requiring the
    # multiprocessing-safe entrypoint guard.
    out = io.StringIO()
    success = True
    error = None
    result = None
    plots = []
    plots_base64 = []

    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            env = {"__name__": "__main__"}
            code_obj, result_var = _prepare_code(code)
            exec(code_obj, env, env)
            if result_var and result_var in env:
                result = env[result_var]
            if save_plots:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    for num in plt.get_fignums():
                        fig = plt.figure(num)
                        path = os.path.join(temp_dir, f"plot_{num}.png")
                        fig.savefig(path)
                        plots.append(path)
                        try:
                            # Base64 for browser-safe rendering.
                            with open(path, "rb") as image_file:
                                encoded = base64.b64encode(image_file.read()).decode("utf-8")
                            plots_base64.append(encoded)
                        except Exception:
                            plots_base64.append("")
                    plt.close("all")
                except Exception:
                    pass
        except Exception:
            success = False
            error = traceback.format_exc()

    res = {
        "success": success,
        "output": pip_output + out.getvalue(),
        "result": result,
        "plots": plots,
        "plots_base64": plots_base64,
        "error": error,
        "installed_packages": installed_packages,
    }
    return res
