#!/usr/bin/env python3
"""
Hytale Toolkit - Installation Wizard

A comprehensive setup script that guides you through:
1. Selecting your Hytale installation
2. Decompiling server source code
3. Generating Javadocs (optional)
4. Setting up the RAG database
5. Configuring Claude Code integration
"""

__version__ = "1.0.0"

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path


def get_base_path() -> Path:
    """Get the base path for resources, handling both normal and frozen (PyInstaller) execution."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return Path(sys.executable).parent
    else:
        # Running as script
        return Path(__file__).parent.resolve()


def get_repo_root() -> Path:
    """Get the repository root path."""
    if getattr(sys, 'frozen', False):
        # When frozen, we're in the same directory as the executable
        return get_base_path()
    else:
        return get_base_path().parent


# Add parent directory to path for shared utilities (only when not frozen)
if not getattr(sys, 'frozen', False):
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
    from logger import setup_logging, log_command, log_exception, log_section
else:
    # When frozen, use bundled logger or create a simple fallback
    try:
        from logger import setup_logging, log_command, log_exception, log_section
    except ImportError:
        # Fallback if logger not bundled
        def setup_logging(name, path):
            import logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(name), None
        def log_command(*args, **kwargs): pass
        def log_exception(*args, **kwargs): pass
        def log_section(*args, **kwargs): pass

# Global logger (initialized in main)
log = None
log_file = None

# ============================================================================
#  Configuration
# ============================================================================

GITHUB_REPO = "logan-mcduffie/Hytale-Toolkit"
OLLAMA_MODEL = "nomic-embed-text"
JDK_VERSION = 21  # LTS version for javadoc generation

SCRIPT_DIR = get_base_path()
REPO_ROOT = get_repo_root()
VINEFLOWER_JAR = REPO_ROOT / "tools" / "vineflower.jar"
ENV_FILE = SCRIPT_DIR / ".env"
DECOMPILED_DIR = REPO_ROOT / "decompiled"  # Always in repo root
JAVADOCS_DIR = REPO_ROOT / "javadocs"      # Also in repo root

# Required contents of a valid Hytale installation
REQUIRED_CONTENTS = ["Client", "Server", "Assets.zip"]

# Default Hytale installation path (Windows only)
DEFAULT_HYTALE_PATH = Path(os.environ.get("APPDATA", "")) / "Hytale" / "install" / "release" / "package" / "game" / "latest"

# Data tables
DATA_TABLES = ["hytale_methods.lance", "hytale_client_ui.lance", "hytale_gamedata.lance", "hytale_docs.lance"]


# ============================================================================
#  Utilities
# ============================================================================

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if platform.system() == 'Windows' else 'clear')


def print_header(title: str):
    """Print a formatted section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def print_step(step: int, total: int, title: str):
    """Print a step header."""
    print()
    print("-" * 60)
    print(f"  Step {step}/{total}: {title}")
    print("-" * 60)
    print()


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt for yes/no input with a default."""
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        response = input(f"  {question} {hint}: ").strip().lower()
        if response == "":
            if log:
                log.info(f"User prompt: '{question}' -> {'Yes' if default else 'No'} (default)")
            return default
        if response in ("y", "yes"):
            if log:
                log.info(f"User prompt: '{question}' -> Yes")
            return True
        if response in ("n", "no"):
            if log:
                log.info(f"User prompt: '{question}' -> No")
            return False
        print("  Please enter 'y' or 'n'.")


def prompt_choice(options: list[tuple[str, str]], prompt_text: str = "Select an option") -> int:
    """Prompt user to select from numbered options. Returns 0-based index."""
    for i, (label, desc) in enumerate(options, 1):
        print(f"  [{i}] {label}")
        if desc:
            for line in desc.split("\n"):
                print(f"      {line}")
        print()

    while True:
        try:
            choice = input(f"  {prompt_text} [1-{len(options)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                if log:
                    log.info(f"User choice: '{prompt_text}' -> [{idx + 1}] {options[idx][0]}")
                return idx
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def prompt_ram_allocation(default: int = 8) -> int:
    """Prompt user for RAM allocation in GB. Returns the value in GB."""
    print(f"  How much RAM (in GB) should be allocated for decompilation/docs?")
    print(f"  Default is {default}GB which is usually enough.")
    print(f"  If you encounter OutOfMemory errors, try increasing this value.")
    print()

    while True:
        response = input(f"  RAM allocation in GB [{default}]: ").strip()
        if response == "":
            if log:
                log.info(f"User input: RAM allocation -> {default}GB (default)")
            return default
        try:
            value = int(response)
            if 2 <= value <= 64:
                if log:
                    log.info(f"User input: RAM allocation -> {value}GB")
                return value
            print("  Please enter a value between 2 and 64 GB.")
        except ValueError:
            print("  Please enter a valid number.")


def is_admin() -> bool:
    """Check if running with admin/root privileges."""
    if platform.system() == "Windows":
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False
    return os.geteuid() == 0


def command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    if platform.system() == "Windows":
        result = subprocess.run(["where", cmd], capture_output=True, shell=True)
    else:
        result = subprocess.run(["which", cmd], capture_output=True)
    return result.returncode == 0


def run_command(cmd: list[str], cwd: Path = None, env: dict = None, shell: bool = None) -> tuple[int, str]:
    """Run a command and return exit code and output."""
    try:
        use_shell = shell if shell is not None else (platform.system() == "Windows")
        cmd_for_log = cmd
        if use_shell and isinstance(cmd, list):
            cmd = " ".join(cmd)

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, encoding='utf-8', errors='replace', shell=use_shell, env=merged_env
        )
        output = result.stdout + result.stderr

        # Log the command execution
        if log:
            log_command(log, cmd_for_log, result.returncode, output, cwd)

        return result.returncode, output
    except Exception as e:
        if log:
            log.error(f"Command failed with exception: {e}")
            log_exception(log, "run_command")
        return 1, str(e)


# ============================================================================
#  .env File Management
# ============================================================================

def load_env() -> dict[str, str]:
    """Load existing .env file."""
    env = {}
    if ENV_FILE.exists():
        with open(ENV_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip()
                    # Strip surrounding quotes and unescape
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1].replace('\\"', '"')
                    env[key.strip()] = value
    return env


def save_env(env: dict[str, str]):
    """Save .env file."""
    with open(ENV_FILE, "w") as f:
        for key, value in env.items():
            # Quote values that contain spaces or special characters
            if ' ' in value or '"' in value or "'" in value:
                # Escape any existing double quotes and wrap in double quotes
                escaped = value.replace('"', '\\"')
                f.write(f'{key}="{escaped}"\n')
            else:
                f.write(f"{key}={value}\n")


# ============================================================================
#  Folder Picker & Validation
# ============================================================================

def open_folder_picker(title: str = "Select Folder") -> str | None:
    """Open a native folder picker dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        folder = filedialog.askdirectory(title=title)
        root.destroy()

        return folder if folder else None
    except ImportError:
        return None


def validate_hytale_installation(folder: str) -> tuple[bool, list[str]]:
    """
    Validate that a folder is a valid Hytale installation.
    Returns (is_valid, missing_items).
    """
    folder_path = Path(folder)

    if log:
        log.debug(f"Validating Hytale installation at: {folder}")

    if not folder_path.exists():
        if log:
            log.warning(f"Folder does not exist: {folder}")
        return False, ["Folder does not exist"]

    missing = []
    for item in REQUIRED_CONTENTS:
        item_path = folder_path / item
        exists = item_path.exists()
        if log:
            log.debug(f"  Checking {item}: {'found' if exists else 'MISSING'}")
        if not exists:
            missing.append(item)

    is_valid = len(missing) == 0
    if log:
        if is_valid:
            log.info(f"Valid Hytale installation found at: {folder}")
        else:
            log.warning(f"Invalid installation - missing: {missing}")

    return is_valid, missing


def get_hytale_install_path(env: dict[str, str]) -> str | None:
    """Get Hytale install path from env or prompt user."""

    # Check for existing saved path
    existing = env.get("HYTALE_INSTALL_PATH", "")
    if existing:
        is_valid, missing = validate_hytale_installation(existing)
        if is_valid:
            print(f"  Found saved Hytale installation: {existing}")
            if prompt_yes_no("Use this path?", default=True):
                return existing
        else:
            print(f"  Saved path is no longer valid (missing: {', '.join(missing)})")

    # Check default Hytale installation path (Windows)
    if platform.system() == "Windows" and DEFAULT_HYTALE_PATH.exists():
        default_str = str(DEFAULT_HYTALE_PATH)
        is_valid, missing = validate_hytale_installation(default_str)
        if is_valid:
            print(f"\n  Found Hytale at default location: {default_str}")
            if prompt_yes_no("Use this path?", default=True):
                return default_str

    # Show instructions
    print()
    print("  Please select your Hytale installation folder.")
    print()
    print("  The path should look like:")
    print("    <drive>:\\...\\Roaming\\install\\release\\package\\game\\latest")
    print()
    print("  The folder MUST contain:")
    print("    - Client\\       (client data files)")
    print("    - Server\\       (HytaleServer.jar)")
    print("    - Assets.zip    (game assets)")
    print()

    max_attempts = 3
    for attempt in range(max_attempts):
        if attempt > 0:
            print(f"\n  Attempt {attempt + 1}/{max_attempts}")

        print("  A folder picker will open. If it doesn't appear, you can type the path.")
        input("  Press Enter to open folder picker...")

        folder = open_folder_picker("Select Hytale Installation Folder (the 'latest' folder)")

        if not folder:
            print()
            folder = input("  Enter path manually: ").strip().strip('"').strip("'")

        if not folder:
            print("  No path provided.")
            continue

        # Validate
        is_valid, missing = validate_hytale_installation(folder)

        if is_valid:
            print(f"\n  Valid installation found!")
            print(f"    Client/     : Found")
            print(f"    Server/     : Found")
            print(f"    Assets.zip  : Found")
            if log:
                log.info(f"User selected Hytale installation path: {folder}")
            return folder
        else:
            print(f"\n  ERROR: Invalid Hytale installation folder.")
            print(f"  Missing: {', '.join(missing)}")
            print()
            print("  Make sure you selected the 'latest' folder that contains")
            print("  Client/, Server/, and Assets.zip")

            if attempt < max_attempts - 1:
                if not prompt_yes_no("Try again?", default=True):
                    return None

    print("\n  Maximum attempts reached.")
    return None


# ============================================================================
#  Decompilation
# ============================================================================

def decompile_server(install_path: str, env: dict[str, str], ram_gb: int = 8) -> bool:
    """Decompile HytaleServer.jar using Vineflower."""
    if log:
        log_section(log, "Decompilation")
        log.info(f"Starting decompilation with {ram_gb}GB RAM")

    server_jar = Path(install_path) / "Server" / "HytaleServer.jar"

    if not server_jar.exists():
        if log:
            log.error(f"HytaleServer.jar not found at {server_jar}")
        print(f"  ERROR: HytaleServer.jar not found at {server_jar}")
        return False

    if not VINEFLOWER_JAR.exists():
        if log:
            log.error(f"Vineflower not found at {VINEFLOWER_JAR}")
        print(f"  ERROR: Vineflower not found at {VINEFLOWER_JAR}")
        return False

    # Check for existing decompiled code
    if DECOMPILED_DIR.exists() and any(DECOMPILED_DIR.iterdir()):
        print(f"  Decompiled code already exists at: {DECOMPILED_DIR}")
        if not prompt_yes_no("Re-decompile? (This will delete existing files)", default=False):
            return True
        shutil.rmtree(DECOMPILED_DIR)

    DECOMPILED_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print(f"  Source:  {server_jar}")
    print(f"  Output:  {DECOMPILED_DIR}")
    print(f"  RAM:     {ram_gb}GB")
    print()
    print("  Decompiling... (this may take several minutes)")
    print()

    # Run Vineflower with progress output
    # Cap threads to avoid OOM - Vineflower defaults to 16 threads, each using significant heap
    thread_count = max(1, min(4, ram_gb // 2))
    cmd = [
        "java",
        "-Xms2G",           # Initial heap size
        f"-Xmx{ram_gb}G",   # Maximum heap size (user-configured)
        "-jar", str(VINEFLOWER_JAR),
        f"-thr={thread_count}",  # Limit threads to prevent OOM
        "-dgs=1",  # Decompile generic signatures
        "-asc=1",  # ASCII string characters
        "-rsy=1",  # Remove synthetic class members
        str(server_jar),
        str(DECOMPILED_DIR)
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )

        # Show progress with single-line updates
        class_count = 0
        for line in process.stdout:
            line = line.strip()
            if line:
                # Handle "INFO:  Loading Class:" or "Decompiling class" lines
                if "Loading Class:" in line or "Decompiling class" in line:
                    class_count += 1
                    # Extract class name from the line
                    if "Loading Class:" in line:
                        class_name = line.split("Loading Class:")[-1].split()[0] if "Loading Class:" in line else ""
                    else:
                        class_name = line.split()[-1] if line.split() else ""
                    short_name = class_name.split("/")[-1] if "/" in class_name else class_name
                    # Overwrite the same line with progress
                    print(f"\r  [{class_count:,} classes] {short_name[:45]:<45}", end="", flush=True)
                elif "error" in line.lower() and "info:" not in line.lower():
                    # Skip benign decompiler warnings
                    if any(skip in line for skip in [
                        "Cannot copy entry META-INF",
                        "cannot be decomposed",
                        "Unable to simplify switch",
                    ]):
                        continue
                    # Only show actual errors, not INFO lines containing 'error' in class names
                    print(f"\n  {line}")

        process.wait()
        print()  # New line after progress

        if process.returncode == 0:
            if log:
                log.info(f"Decompilation completed successfully ({class_count} classes)")
            print("  Decompilation complete!")
            # Fix decompilation artifacts for javadoc compatibility
            fix_decompiled_files()
            return True
        else:
            if log:
                log.error(f"Decompilation failed with exit code {process.returncode}")
            print(f"  Decompilation failed with exit code {process.returncode}")
            return False

    except FileNotFoundError:
        if log:
            log.error("Java not found in PATH")
            log_exception(log, "decompile_server")
        print("\n  ERROR: Java not found. Please install Java and try again.")
        return False
    except Exception as e:
        if log:
            log.error(f"Decompilation exception: {e}")
            log_exception(log, "decompile_server")
        print(f"\n  ERROR: {e}")
        return False


# ============================================================================
#  Fix Decompilation Artifacts
# ============================================================================

def fix_decompiled_file(filepath: Path) -> bool:
    """
    Fix a single Java file with decompilation artifacts.
    Returns True if modifications were made.
    """
    try:
        content = filepath.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return False

    original = content

    # Replace <unrepresentable> with a valid Java identifier
    # This token appears where the decompiler couldn't resolve a class name
    content = content.replace('<unrepresentable>', 'DecompilerPlaceholder')

    # Replace $assertionsDisabled references with false
    # These are assertion-related checks that aren't meaningful in decompiled code
    content = re.sub(r'DecompilerPlaceholder\.\$assertionsDisabled', 'false', content)

    # Fix interfaces with static initializer blocks (not valid Java)
    if re.search(r'^public\s+interface\s+\w+', content, re.MULTILINE):
        # Find CODEC field declaration without initialization
        codec_match = re.search(r'(BuilderCodecMapCodec<[^>]+>)\s+CODEC\s*;', content)
        if codec_match:
            # Find the initialization in the static block
            init_match = re.search(r'CODEC\s*=\s*(new\s+BuilderCodecMapCodec<>\([^)]*\))\s*;', content)
            if init_match:
                # Replace uninitialized field with initialized one
                content = re.sub(
                    r'(BuilderCodecMapCodec<[^>]+>)\s+CODEC\s*;',
                    f'\\1 CODEC = {init_match.group(1)};',
                    content
                )

        # Remove static blocks from interfaces (they're not allowed)
        lines = content.split('\n')
        new_lines = []
        in_static_block = False
        brace_count = 0

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('static {') or stripped == 'static {':
                in_static_block = True
                brace_count = line.count('{') - line.count('}')
                continue

            if in_static_block:
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    in_static_block = False
                continue

            new_lines.append(line)

        content = '\n'.join(new_lines)

    if content != original:
        try:
            filepath.write_text(content, encoding='utf-8')
            return True
        except Exception:
            return False

    return False


def fix_decompiled_files() -> int:
    """
    Fix decompilation artifacts in all Java files.
    Returns the number of files fixed.
    """
    if not DECOMPILED_DIR.exists():
        return 0

    print("  Fixing decompilation artifacts...")

    count = 0
    java_files = list(DECOMPILED_DIR.rglob("*.java"))
    total = len(java_files)

    for i, filepath in enumerate(java_files):
        if fix_decompiled_file(filepath):
            count += 1
        # Show progress every 100 files
        if (i + 1) % 100 == 0:
            print(f"\r  Processed {i + 1}/{total} files ({count} fixed)...", end="", flush=True)

    print(f"\r  Fixed {count} files with decompilation artifacts.    ")
    return count


# ============================================================================
#  Javadoc Generation
# ============================================================================

def generate_javadocs(include_private: bool = False, ram_gb: int = 8, javadoc_path: str = "javadoc") -> bool:
    """Generate Javadocs from decompiled source."""
    if not DECOMPILED_DIR.exists():
        print("  ERROR: No decompiled code found. Please decompile first.")
        return False

    if JAVADOCS_DIR.exists():
        if not prompt_yes_no("Javadocs already exist. Regenerate?", default=False):
            return True
        shutil.rmtree(JAVADOCS_DIR)

    JAVADOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure decompilation artifacts are fixed before javadoc
    fix_decompiled_files()

    print()
    print(f"  Generating Javadocs (using {ram_gb}GB RAM)... (this may take a while)")
    print()

    # Find Hytale Java files only (com/hypixel/**), excluding third-party libs
    # Also exclude package-info.java (malformed from decompilation)
    hytale_src = DECOMPILED_DIR / "com" / "hypixel"
    if not hytale_src.exists():
        print(f"  ERROR: Hytale source not found at {hytale_src}")
        return False
    java_files = [f for f in hytale_src.rglob("*.java") if f.name != "package-info.java"]
    print(f"  Found {len(java_files)} Hytale Java files (excluding third-party libs)")

    # Build javadoc command (no -quiet so we can show progress)
    cmd = [
        javadoc_path,
        "-J-Xms2G",             # Initial heap size
        f"-J-Xmx{ram_gb}G",     # Maximum heap size (user-configured)
        "-d", str(JAVADOCS_DIR),
        "-Xdoclint:none",       # Suppress documentation warnings
        "--ignore-source-errors",  # Continue despite syntax errors in decompiled code
    ]

    if include_private:
        cmd.append("-private")

    # Add source files (via argfile to avoid command line length limits)
    # Use forward slashes - Java's @argfile parser treats backslashes as escape chars
    argfile = SCRIPT_DIR / ".javadoc-files.txt"
    with open(argfile, "w") as f:
        for java_file in java_files:
            path_str = str(java_file).replace("\\", "/")
            f.write(f'"{path_str}"\n')

    cmd.append(f"@{argfile}")

    try:
        # Use Popen to stream output and show progress
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )

        # Third-party packages to suppress errors for (missing imports are expected)
        suppressed_packages = [
            "io.sentry", "io.netty", "it.unimi.dsi", "javax.annotation",
            "com.google.common", "org.bson", "ch.randelshofer", "joptsimple",
            "org.bouncycastle", "com.github.luben", "com.nimbusds", "org.slf4j",
        ]

        file_count = 0
        gen_count = 0
        error_count = 0
        total_files = len(java_files)
        # Estimate ~1.5x files for generation (classes + packages + index files)
        estimated_gen_total = int(total_files * 1.5)

        for line in process.stdout:
            line = line.strip()
            if line:
                # Count "Loading source file" lines for progress
                if "Loading source file" in line:
                    file_count += 1
                    # Extract just the class name
                    parts = line.split("Loading source file")[-1].strip()
                    class_name = parts.split("/")[-1].replace(".java...", "")
                    percent = (file_count * 100) // total_files
                    print(f"\r  [{percent:3d}%] Loading: {class_name[:50]:<50}", end="", flush=True)
                elif "Generating" in line:
                    # Show generation progress
                    gen_count += 1
                    percent = min(99, (gen_count * 100) // estimated_gen_total)
                    # Extract the file being generated
                    gen_file = line.split("/")[-1].replace("...", "") if "/" in line else "docs"
                    print(f"\r  [{percent:3d}%] Generating: {gen_file[:50]:<50}", end="", flush=True)
                elif "error:" in line.lower() and "warning" not in line.lower():
                    # Suppress errors about missing third-party packages
                    # Also suppress "cannot find symbol" errors (cascading from missing imports)
                    if any(pkg in line for pkg in suppressed_packages):
                        error_count += 1
                    elif "cannot find symbol" in line or "package" in line and "does not exist" in line:
                        error_count += 1
                    else:
                        # Show non-suppressed errors
                        print(f"\n  {line}")

        process.wait()
        print(f"\r  [100%] Done!{' ' * 60}")  # Clear line and show completion

        if error_count > 0:
            print(f"  (Suppressed {error_count} missing third-party import errors)")

        # Check if any output was actually generated
        index_html = JAVADOCS_DIR / "index.html"
        generated_files = list(JAVADOCS_DIR.glob("**/*.html"))

        if index_html.exists():
            argfile.unlink()  # Clean up on success
            print(f"  Javadocs generated successfully!")
            print(f"  Generated {len(generated_files)} HTML files.")
            print(f"  Open {index_html} in a browser to view.")
            return True
        elif generated_files:
            argfile.unlink()  # Clean up if we got partial output
            print("  Javadoc generation completed with warnings.")
            print(f"  Generated {len(generated_files)} HTML files.")
            return True
        else:
            print("  ERROR: Javadoc failed to generate any output.")
            print(f"  Argfile preserved at: {argfile}")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


# ============================================================================
#  Node.js / npm Setup
# ============================================================================

def check_npm_installed() -> bool:
    """Check if npm is installed."""
    return command_exists("npm")


def check_node_installed() -> bool:
    """Check if Node.js is installed."""
    return command_exists("node")


def install_nodejs() -> bool:
    """Install Node.js (which includes npm) based on the current OS."""
    system = platform.system()

    if system == "Windows":
        if not is_admin():
            print("\n  ERROR: Administrator privileges required to install Node.js.")
            print("  Please restart this script as Administrator, or install Node.js manually:")
            print("    https://nodejs.org/")
            return False
        print("  Installing Node.js via winget...", flush=True)
        exit_code, output = run_command([
            "winget", "install", "-e", "--id", "OpenJS.NodeJS.LTS",
            "--accept-source-agreements", "--accept-package-agreements"
        ])
        if exit_code != 0:
            print(f"  Installation failed. Please install manually: https://nodejs.org/")
            return False

    elif system == "Darwin":  # macOS
        if command_exists("brew"):
            print("  Installing Node.js via Homebrew...", flush=True)
            exit_code, output = run_command(["brew", "install", "node"], shell=False)
            if exit_code != 0:
                print(f"  Installation failed. Please install manually: https://nodejs.org/")
                return False
        else:
            print("  Homebrew not found. Please install Node.js manually:")
            print("    https://nodejs.org/")
            print("  Or install Homebrew first: https://brew.sh")
            return False

    elif system == "Linux":
        print("  Installing Node.js...", flush=True)
        # Use NodeSource setup script for latest LTS
        exit_code, output = run_command(
            ["bash", "-c", "curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && sudo apt-get install -y nodejs"],
            shell=False
        )
        if exit_code != 0:
            # Try alternative for non-Debian systems
            print("  Debian-based install failed, trying alternative...")
            exit_code, output = run_command(
                ["bash", "-c", "curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash - && sudo yum install -y nodejs"],
                shell=False
            )
            if exit_code != 0:
                print(f"  Installation failed. Please install manually: https://nodejs.org/")
                return False

    else:
        print(f"  Unsupported OS: {system}")
        print("  Please install Node.js manually: https://nodejs.org/")
        return False

    print("  Node.js installed!")
    return True


def setup_npm() -> bool:
    """Ensure npm is installed, installing Node.js if needed."""
    print("  Checking for npm...", flush=True)

    if check_npm_installed():
        # Get version for display
        exit_code, output = run_command(["npm", "--version"])
        version = output.strip() if exit_code == 0 else "unknown"
        print(f"  npm is installed (v{version}).")
        return True

    print("  npm is not installed.")
    print()
    print("  npm is required to run the MCP server. It comes bundled with Node.js.")
    print()

    if not prompt_yes_no("Install Node.js (includes npm)?", default=True):
        print("  Please install Node.js manually: https://nodejs.org/")
        return False

    if not install_nodejs():
        return False

    # Verify installation worked
    # On Windows, we may need to refresh the PATH
    if platform.system() == "Windows":
        print("  Verifying installation...", flush=True)
        # Try to find npm in common install locations
        possible_paths = [
            Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "nodejs",
            Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "nodejs",
            Path.home() / "AppData" / "Roaming" / "npm",
        ]
        for npm_dir in possible_paths:
            npm_exe = npm_dir / "npm.cmd"
            if npm_exe.exists():
                print(f"  Found npm at: {npm_dir}")
                print()
                print("  NOTE: You may need to restart your terminal for npm to be in your PATH.")
                print("  If the next step fails, please close and reopen your terminal,")
                print("  then run setup.py again.")
                print()
                return True

    if check_npm_installed():
        print("  npm is now available!")
        return True

    print("  WARNING: npm was installed but may not be in PATH yet.")
    print("  Please restart your terminal and run setup.py again.")
    return False


# ============================================================================
#  Ollama Setup
# ============================================================================

def check_ollama_installed() -> bool:
    return command_exists("ollama")


def check_ollama_running() -> bool:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except:
        return False


def check_ollama_model_available(model: str = OLLAMA_MODEL) -> bool:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            models = data.get("models", [])
            return any(m["name"] == model or m["name"].startswith(f"{model}:") for m in models)
    except:
        return False


def install_ollama() -> bool:
    """Install Ollama based on the current OS."""
    system = platform.system()

    if system == "Windows":
        if not is_admin():
            print("\n  ERROR: Administrator privileges required to install Ollama.")
            print("  Please restart this script as Administrator.")
            return False
        print("  Installing Ollama via winget...", flush=True)
        exit_code, output = run_command([
            "winget", "install", "-e", "--id", "Ollama.Ollama",
            "--accept-source-agreements", "--accept-package-agreements"
        ])
        if exit_code != 0:
            print(f"  Installation failed. Please install manually: https://ollama.com/download")
            return False

    elif system == "Darwin":  # macOS
        if not command_exists("brew"):
            print("  Homebrew is required. Install from: https://brew.sh")
            return False
        print("  Installing Ollama via Homebrew...", flush=True)
        exit_code, output = run_command(["brew", "install", "ollama"], shell=False)
        if exit_code != 0:
            print(f"  Installation failed. Please install manually: https://ollama.com/download")
            return False

    elif system == "Linux":
        print("  Installing Ollama...", flush=True)
        exit_code, output = run_command(
            ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
            shell=False
        )
        if exit_code != 0:
            print(f"  Installation failed. Please install manually: https://ollama.com/download")
            return False

    else:
        print(f"  Unsupported OS: {system}")
        return False

    print("  Ollama installed!")
    return True


def start_ollama() -> bool:
    """Start Ollama server in the background."""
    system = platform.system()

    if system == "Windows":
        subprocess.Popen(
            "ollama serve",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )

    # Wait for startup
    for _ in range(15):
        time.sleep(1)
        if check_ollama_running():
            return True
    return False


def pull_ollama_model(model: str = OLLAMA_MODEL) -> bool:
    """Pull the required embedding model with progress."""
    print(f"  Downloading {model} model...", flush=True)
    print()

    try:
        cmd = ["ollama", "pull", model]
        use_shell = platform.system() == "Windows"
        process = subprocess.Popen(
            " ".join(cmd) if use_shell else cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding='utf-8',
            errors='replace',
            shell=use_shell
        )

        for line in process.stdout:
            line = line.strip()
            if line:
                print(f"  {line}")

        process.wait()
        return process.returncode == 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def setup_ollama() -> bool:
    """Full Ollama setup: install, start, pull model."""
    print("  Checking Ollama...", flush=True)

    if not check_ollama_installed():
        print("  Ollama is not installed.")
        if not prompt_yes_no("Install Ollama?", default=True):
            print("  Please install Ollama manually: https://ollama.com/download")
            return False
        if not install_ollama():
            return False

    print("  Ollama is installed.")

    if not check_ollama_running():
        print("  Starting Ollama server...", flush=True)
        if not start_ollama():
            print("  Failed to start. Please run 'ollama serve' manually.")
            return False
    print("  Ollama server is running.")

    if not check_ollama_model_available(OLLAMA_MODEL):
        print(f"  Model '{OLLAMA_MODEL}' not found.")
        if not pull_ollama_model(OLLAMA_MODEL):
            return False
    print(f"  Model '{OLLAMA_MODEL}' is ready.")

    return True


# ============================================================================
#  JDK Setup (for Javadoc generation)
# ============================================================================

def detect_jdk_path(version: int) -> str | None:
    """Try to detect JDK installation path for a specific version."""
    system = platform.system()
    version_str = str(version)
    possible_paths = []

    if system == "Windows":
        possible_paths = [
            Path(f"C:/Program Files/Java/jdk-{version}"),
            Path(f"C:/Program Files/Eclipse Adoptium/jdk-{version}"),
        ]
        # Check JAVA_HOME if it contains the version
        java_home = os.environ.get("JAVA_HOME", "")
        if version_str in java_home:
            possible_paths.insert(0, Path(java_home))

        # Check for versioned directories like jdk-21.0.1
        java_dir = Path("C:/Program Files/Java")
        if java_dir.exists():
            for item in java_dir.iterdir():
                if item.is_dir() and item.name.startswith(f"jdk-{version}"):
                    possible_paths.insert(0, item)
        adoptium_dir = Path("C:/Program Files/Eclipse Adoptium")
        if adoptium_dir.exists():
            for item in adoptium_dir.iterdir():
                if item.is_dir() and f"-{version}" in item.name:
                    possible_paths.insert(0, item)

        # Check local .jdks folder (where we download to)
        local_jdks = SCRIPT_DIR / ".jdks"
        if local_jdks.exists():
            for item in local_jdks.iterdir():
                if item.is_dir() and f"jdk-{version}" in item.name:
                    possible_paths.insert(0, item)

    elif system == "Darwin":  # macOS
        possible_paths = [
            Path(f"/Library/Java/JavaVirtualMachines/jdk-{version}.jdk/Contents/Home"),
            Path(f"/Library/Java/JavaVirtualMachines/temurin-{version}.jdk/Contents/Home"),
        ]
        java_home = os.environ.get("JAVA_HOME", "")
        if version_str in java_home:
            possible_paths.insert(0, Path(java_home))

        # Check local .jdks folder
        local_jdks = SCRIPT_DIR / ".jdks"
        if local_jdks.exists():
            for item in local_jdks.iterdir():
                if item.is_dir() and f"jdk-{version}" in item.name:
                    # macOS extracted JDKs have Contents/Home structure
                    home_path = item / "Contents" / "Home"
                    if home_path.exists():
                        possible_paths.insert(0, home_path)
                    else:
                        possible_paths.insert(0, item)

    else:  # Linux
        possible_paths = [
            Path(f"/usr/lib/jvm/java-{version}-openjdk"),
            Path(f"/usr/lib/jvm/jdk-{version}"),
            Path(f"/usr/lib/jvm/temurin-{version}"),
        ]
        java_home = os.environ.get("JAVA_HOME", "")
        if version_str in java_home:
            possible_paths.insert(0, Path(java_home))

        # Check local .jdks folder
        local_jdks = SCRIPT_DIR / ".jdks"
        if local_jdks.exists():
            for item in local_jdks.iterdir():
                if item.is_dir() and f"jdk-{version}" in item.name:
                    possible_paths.insert(0, item)

    javadoc_exe = "javadoc.exe" if system == "Windows" else "javadoc"
    for path in possible_paths:
        if path and path.exists() and (path / "bin" / javadoc_exe).exists():
            return str(path)

    return None


def get_adoptium_download_url(version: int) -> tuple[str, str] | None:
    """Get the Adoptium download URL for a specific JDK version.

    Returns (url, filename) or None if not available.
    """
    system = platform.system()
    machine = platform.machine().lower()

    # Map platform to Adoptium API values
    if system == "Windows":
        os_name = "windows"
    elif system == "Darwin":
        os_name = "mac"
    else:
        os_name = "linux"

    # Map architecture
    if machine in ("x86_64", "amd64"):
        arch = "x64"
    elif machine in ("aarch64", "arm64"):
        arch = "aarch64"
    else:
        arch = "x64"  # Default fallback

    # Try to get the latest release info from Adoptium API
    api_url = f"https://api.adoptium.net/v3/assets/latest/{version}/ga"
    params = f"?architecture={arch}&image_type=jdk&os={os_name}&vendor=eclipse"

    try:
        # Create SSL context to handle macOS certificate issues
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(api_url + params)
        req.add_header("Accept", "application/json")

        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            data = json.loads(response.read().decode())
            if data and len(data) > 0:
                binary = data[0].get("binary", {})
                package = binary.get("package", {})
                download_url = package.get("link")
                filename = package.get("name")

                if download_url and filename:
                    return (download_url, filename)
    except Exception as e:
        print(f"    Warning: Could not query Adoptium API: {e}")

    return None


def download_and_extract_jdk(version: int, target_dir: Path) -> str | None:
    """Download and extract a JDK from Adoptium.

    Returns the path to the extracted JDK, or None on failure.
    """
    print(f"  Fetching JDK {version} download information...")

    url_info = get_adoptium_download_url(version)
    if not url_info:
        print(f"    ERROR: Could not find JDK {version} download")
        return None

    download_url, filename = url_info
    print(f"    Found: {filename}")

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / filename

    # Download the archive
    print(f"  Downloading JDK {version}...")
    print(f"    This may take a few minutes...")

    try:
        # Use urlopen with SSL context (urlretrieve doesn't support SSL context)
        # This fixes SSL certificate issues on macOS
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(download_url, headers={"User-Agent": "Hytale-RAG-Setup"})
        with urllib.request.urlopen(req, context=ctx, timeout=600) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 8192

            with open(archive_path, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = min(100, downloaded * 100 // total_size)
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r    [{percent:3d}%] {mb_downloaded:.1f} / {mb_total:.1f} MB", end="", flush=True)

        print()  # Newline after progress
    except Exception as e:
        print(f"\n    ERROR: Download failed: {e}")
        return None

    # Extract the archive
    print(f"  Extracting JDK {version}...")

    try:
        if filename.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        elif filename.endswith(".tar.gz") or filename.endswith(".tgz"):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(target_dir)
        else:
            print(f"    ERROR: Unknown archive format: {filename}")
            return None

        # Remove the archive after extraction
        archive_path.unlink()

        # Find the extracted JDK directory
        for item in target_dir.iterdir():
            if item.is_dir() and f"jdk-{version}" in item.name:
                # On macOS, the JDK might be in Contents/Home
                if platform.system() == "Darwin":
                    home_path = item / "Contents" / "Home"
                    if home_path.exists():
                        print(f"    Extracted to: {home_path}")
                        return str(home_path)
                print(f"    Extracted to: {item}")
                return str(item)

        print("    ERROR: Could not find extracted JDK directory")
        return None

    except Exception as e:
        print(f"    ERROR: Extraction failed: {e}")
        return None


def get_jdk_javadoc_path() -> str | None:
    """Get the path to javadoc, downloading JDK if needed.

    Returns the full path to javadoc executable, or None if unavailable.
    """
    # First check if javadoc is already in PATH
    if command_exists("javadoc"):
        return "javadoc"  # Use system javadoc

    # Try to detect installed JDK
    jdk_path = detect_jdk_path(JDK_VERSION)
    if jdk_path:
        javadoc_exe = "javadoc.exe" if platform.system() == "Windows" else "javadoc"
        return str(Path(jdk_path) / "bin" / javadoc_exe)

    # JDK not found - offer to download
    print(f"  JDK not found. Javadoc requires a Java Development Kit.")
    print()

    options = [
        ("Download JDK automatically", f"Downloads JDK {JDK_VERSION} from Eclipse Adoptium (~200MB)"),
        ("Skip Javadoc generation", "Continue without generating Javadocs"),
    ]

    choice = prompt_choice(options, "How would you like to proceed?")

    if choice == 0:
        # Download JDK
        jdks_dir = SCRIPT_DIR / ".jdks"
        print()
        jdk_path = download_and_extract_jdk(JDK_VERSION, jdks_dir)
        if jdk_path:
            print(f"    JDK {JDK_VERSION} installed successfully!")
            javadoc_exe = "javadoc.exe" if platform.system() == "Windows" else "javadoc"
            return str(Path(jdk_path) / "bin" / javadoc_exe)
        else:
            print("    WARNING: Failed to install JDK")
            return None

    # User chose to skip
    return None


# ============================================================================
#  Database Download & Verification
# ============================================================================

def verify_database_files(lancedb_dir: Path) -> tuple[bool, str]:
    """
    Verify database files exist and appear valid.
    Returns (success, error_message).
    """
    if not lancedb_dir.exists():
        return False, "Database directory does not exist"

    print("  Verifying database files...")
    print()

    # Collect all files to verify
    all_files = []
    for table in DATA_TABLES:
        table_dir = lancedb_dir / table
        if not table_dir.exists():
            return False, f"Missing table: {table}"
        # Collect all files in the table directory
        table_files = list(table_dir.rglob("*"))
        all_files.extend([(table, f) for f in table_files if f.is_file()])

    if not all_files:
        return False, "No files found in database"

    total_files = len(all_files)
    verified = 0
    errors = []
    current_table = ""

    bar_width = 30

    for table, filepath in all_files:
        # Update display when table changes
        if table != current_table:
            current_table = table
            table_name = table.replace(".lance", "")

        # Verify file is readable and non-empty
        try:
            size = filepath.stat().st_size
            if size == 0:
                errors.append(f"Empty file: {filepath.name}")
        except Exception as e:
            errors.append(f"Cannot read {filepath.name}: {e}")

        verified += 1
        percent = verified * 100 / total_files
        filled = int(bar_width * percent / 100)
        bar = "=" * filled + "-" * (bar_width - filled)
        print(f"\r  [{bar}] {percent:5.1f}% ({verified}/{total_files} files) {table_name:<20}", end="", flush=True)

    print()  # New line after progress bar

    if errors:
        print()
        for error in errors[:5]:  # Show first 5 errors
            print(f"  WARNING: {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more warnings")
        print()

    # Check for required manifest files in each table
    for table in DATA_TABLES:
        table_dir = lancedb_dir / table
        # LanceDB tables should have at least a _versions directory or manifest
        has_versions = (table_dir / "_versions").exists()
        has_manifest = any(table_dir.glob("*.manifest"))
        if not has_versions and not has_manifest:
            return False, f"Table {table} appears corrupted (missing manifest)"

    return True, ""


# ============================================================================
#  MCP Client Integration
# ============================================================================

# Import shared MCP configuration functions
from mcp_config import (
    MCP_CLIENTS,
    get_mcp_command_stdio,
    get_mcp_command_simple,
    create_start_scripts as _create_start_scripts,
    setup_claude_code,
    setup_vscode as _setup_vscode_base,
    setup_cursor as _setup_cursor_base,
    setup_windsurf,
    setup_codex,
    setup_jetbrains,
    get_vscode_user_settings_path,
    get_cursor_user_settings_path,
    get_client_config_path,
)


def create_start_scripts(script_dir: Path):
    """Create helper scripts for starting the MCP server (CLI wrapper with output)."""
    _create_start_scripts(script_dir, quiet=False)


def setup_vscode(script_dir: Path) -> bool:
    """Configure VS Code / GitHub Copilot MCP server (CLI version with prompts)."""
    # Ask user about installation scope
    print()
    print("    VS Code MCP Configuration:")
    print()
    scope_options = [
        ("Global (Recommended)", "Works in any VS Code workspace.\nConfig stored in your user settings."),
        ("Workspace only", "Only works when this specific folder is open.\nConfig stored in .vscode/mcp.json"),
    ]
    scope_choice = prompt_choice(scope_options, "Installation scope")
    scope = "global" if scope_choice == 0 else "workspace"

    result = _setup_vscode_base(script_dir, scope=scope, quiet=False)

    if scope == "global":
        print()
        print("    The MCP server is now available globally in VS Code.")
    else:
        print()
        print("    NOTE: This only works when this folder is open in VS Code.")

    print()
    print("    To use with GitHub Copilot:")
    print("      1. Open VS Code 1.102+")
    print("      2. Enable Copilot Agent mode")
    print("      3. The MCP server will be available automatically")
    return result


def setup_cursor(script_dir: Path) -> bool:
    """Configure Cursor MCP server (CLI version with prompts)."""
    # Ask user about installation scope
    print()
    print("    Cursor MCP Configuration:")
    print()
    scope_options = [
        ("Global (Recommended)", "Works in any Cursor workspace.\nConfig stored in your user settings."),
        ("Workspace only", "Only works when this specific folder is open.\nConfig stored in .cursor/mcp.json"),
    ]
    scope_choice = prompt_choice(scope_options, "Installation scope")
    scope = "global" if scope_choice == 0 else "workspace"

    result = _setup_cursor_base(script_dir, scope=scope, quiet=False)

    if scope == "global":
        print()
        print("    The MCP server is now available globally in Cursor.")
    else:
        print()
        print("    NOTE: This only works when this folder is open in Cursor.")

    print()
    print("    To use the MCP server:")
    print("      1. Restart Cursor (or reload the window)")
    print("      2. The hytale-rag tools will be available in chat")
    return result


def prompt_multi_choice(options: list[tuple[str, str]], prompt_text: str = "Select options") -> list[int]:
    """Prompt user to select multiple options from numbered list. Returns list of 0-based indices."""
    for i, (label, desc) in enumerate(options, 1):
        print(f"  [{i}] {label}")
        if desc:
            for line in desc.split("\n"):
                print(f"      {line}")
        print()

    print(f"  [A] All of the above")
    print(f"  [0] Skip / None")
    print()

    while True:
        response = input(f"  {prompt_text} (e.g., 1,2,3 or A for all): ").strip().lower()

        if response == "0" or response == "":
            if log:
                log.info(f"User multi-choice: '{prompt_text}' -> None (skipped)")
            return []

        if response == "a":
            if log:
                selected = [options[i][0] for i in range(len(options))]
                log.info(f"User multi-choice: '{prompt_text}' -> All: {selected}")
            return list(range(len(options)))

        try:
            # Parse comma-separated numbers
            indices = []
            for part in response.replace(" ", "").split(","):
                idx = int(part) - 1
                if 0 <= idx < len(options):
                    indices.append(idx)
                else:
                    raise ValueError()
            if indices:
                result = list(set(indices))  # Remove duplicates
                if log:
                    selected = [options[i][0] for i in result]
                    log.info(f"User multi-choice: '{prompt_text}' -> {selected}")
                return result
        except ValueError:
            pass

        print(f"  Please enter numbers 1-{len(options)} separated by commas, 'A' for all, or '0' to skip.")


def setup_mcp_clients(selected_clients: list[str], script_dir: Path) -> dict[str, bool]:
    """Set up MCP integration for selected clients. Returns dict of client_id -> success."""
    results = {}

    # First, create the start scripts that some clients need
    if any(c in selected_clients for c in ["vscode", "cursor", "windsurf", "codex", "jetbrains"]):
        print("  Creating MCP start scripts...")
        create_start_scripts(script_dir)
        print()

    setup_functions = {
        "claude_code": setup_claude_code,
        "vscode": setup_vscode,
        "cursor": setup_cursor,
        "windsurf": setup_windsurf,
        "codex": setup_codex,
        "jetbrains": setup_jetbrains,
    }

    for client_id in selected_clients:
        client_info = MCP_CLIENTS[client_id]
        print(f"  Setting up {client_info['name']}...", flush=True)

        setup_fn = setup_functions.get(client_id)
        if setup_fn:
            try:
                results[client_id] = setup_fn(script_dir)
            except Exception as e:
                print(f"    ERROR: {e}")
                results[client_id] = False
        else:
            print(f"    Skipped: No setup function for {client_id}")
            results[client_id] = False

        print()

    return results


# ============================================================================
#  Main Installation Wizard
# ============================================================================

def main():
    global log, log_file

    # Initialize logging
    log, log_file = setup_logging("setup", REPO_ROOT)
    log_section(log, "Initialization")
    log.info(f"SCRIPT_DIR: {SCRIPT_DIR}")
    log.info(f"REPO_ROOT: {REPO_ROOT}")
    log.info(f"ENV_FILE: {ENV_FILE}")

    clear_screen()

    print()
    print("  +----------------------------------------------------------+")
    print("  |                                                          |")
    print("  |       Welcome to the Hytale Toolkit Setup Wizard!        |")
    print("  |                                                          |")
    print("  +----------------------------------------------------------+")
    print()
    print("  This wizard will help you set up:")
    print("    - Decompiled Hytale source code")
    print("    - Javadoc documentation (optional)")
    print("    - Semantic code search (RAG)")
    print("    - Claude Code integration (optional)")
    print()

    input("  Press Enter to begin...")

    # Validate we're in the right directory
    if not (SCRIPT_DIR / "package.json").exists():
        print("\n  ERROR: Please run this script from the hytale-rag directory.")
        sys.exit(1)

    env = load_env()
    total_steps = 6

    # =========================================================================
    # Step 1: Hytale Installation Path
    # =========================================================================
    print_step(1, total_steps, "Hytale Installation")
    log_section(log, "Step 1: Hytale Installation")

    install_path = get_hytale_install_path(env)
    if not install_path:
        print("\n  Setup cancelled: No valid installation path provided.")
        sys.exit(1)

    env["HYTALE_INSTALL_PATH"] = install_path

    # Auto-detect and save other paths
    path_obj = Path(install_path)
    
    # Client Data
    client_data = path_obj / "Client" / "Data"
    if client_data.exists():
        env["HYTALE_CLIENT_DATA_DIR"] = str(client_data)
        if log:
            log.info(f"Detected Client Data: {client_data}")
    
    # Assets (root dir contains Assets.zip)
    env["HYTALE_ASSETS_DIR"] = install_path
    
    # Decompiled source
    env["HYTALE_DECOMPILED_DIR"] = str(DECOMPILED_DIR)
    
    save_env(env)
    print(f"\n  Saved configuration to .env:")
    print(f"    - HYTALE_INSTALL_PATH")
    if "HYTALE_CLIENT_DATA_DIR" in env:
        print(f"    - HYTALE_CLIENT_DATA_DIR")
    print(f"    - HYTALE_ASSETS_DIR")
    print(f"    - HYTALE_DECOMPILED_DIR")

    # =========================================================================
    # Step 2: Decompile Source Code
    # =========================================================================
    print_step(2, total_steps, "Decompile Source Code")
    log_section(log, "Step 2: Decompile Source Code")

    print("  Decompiling the server source code gives your AI assistant access to")
    print("  the actual Java implementation. This is highly recommended for")
    print("  accurate code assistance and for your own reference.")
    print()

    # Prompt for RAM allocation (used for both decompile and javadocs)
    ram_gb = 8  # Default
    if prompt_yes_no("Decompile server source code? (recommended)", default=True):
        print()
        ram_gb = prompt_ram_allocation(default=8)
        print()
        if not decompile_server(install_path, env, ram_gb=ram_gb):
            if not (DECOMPILED_DIR.exists() and any(DECOMPILED_DIR.iterdir())):
                print("\n  WARNING: Decompilation failed and no existing code found.")
                if not prompt_yes_no("Continue anyway?", default=False):
                    sys.exit(1)
    else:
        if DECOMPILED_DIR.exists() and any(DECOMPILED_DIR.iterdir()):
            print("  Using existing decompiled code.")
        else:
            print("  Skipping decompilation. You can run index-all.py later to decompile.")

    # =========================================================================
    # Step 3: Generate Javadocs
    # =========================================================================
    print_step(3, total_steps, "Generate Javadocs (Optional)")
    log_section(log, "Step 3: Generate Javadocs")

    print("  Javadocs provide browsable HTML documentation of all classes")
    print("  and methods. Useful for exploring the codebase in a browser.")
    print()

    if DECOMPILED_DIR.exists() and any(DECOMPILED_DIR.iterdir()):
        if prompt_yes_no("Generate Javadocs?", default=False):
            print()
            # Check for JDK / offer to download
            javadoc_path = get_jdk_javadoc_path()
            if javadoc_path:
                print()
                include_private = prompt_yes_no(
                    "Include private methods? (Much more verbose, 2-3x larger)",
                    default=False
                )
                generate_javadocs(include_private, ram_gb=ram_gb, javadoc_path=javadoc_path)
            else:
                print("  Skipping Javadoc generation.")
    else:
        print("  Skipped: No decompiled code available.")

    # =========================================================================
    # Step 4: Select Embedding Provider
    # =========================================================================
    print_step(4, total_steps, "Embedding Provider")
    log_section(log, "Step 4: Embedding Provider")

    print("  The RAG system uses embeddings to enable semantic code search.")
    print("  Choose how you want to generate these embeddings:")
    print()

    provider_options = [
        ("Voyage AI (Recommended)",
         "Cloud-based, high quality embeddings.\n"
         "Free tier with 3 queries/min unless you add a payment method.\n"
         "Extremely generous limits - you're unlikely to ever be charged.\n"
         "You can set a $0 spending limit on their dashboard.\n"
         "Best choice if you have slower hardware."),
        ("Ollama (Local)",
         "Runs entirely on your machine, no API key needed.\n"
         "Requires ~2GB disk space for the model. No rate limits.\n"
         "Still very good quality, just slightly below Voyage.")
    ]

    provider_idx = prompt_choice(provider_options, "Select provider")
    provider = "voyage" if provider_idx == 0 else "ollama"
    log.info(f"Selected embedding provider: {provider}")

    env["EMBEDDING_PROVIDER"] = provider

    if provider == "voyage":
        # Clean up Ollama config if switching from Ollama
        if "OLLAMA_MODEL" in env:
            del env["OLLAMA_MODEL"]
            if log:
                log.info("Removed stale OLLAMA_MODEL from env (switched to Voyage)")

        print()
        existing_key = env.get("VOYAGE_API_KEY", "")
        if existing_key:
            print(f"  Found existing API key.")
            if not prompt_yes_no("Use existing key?", default=True):
                existing_key = ""

        if not existing_key:
            print()
            print("  Get a free API key at: https://www.voyageai.com/")
            print("  (You can set a $0 spending limit in their dashboard)")
            print()
            api_key = input("  Enter your Voyage API key: ").strip()
            if not api_key:
                print("  ERROR: API key is required for Voyage AI.")
                sys.exit(1)
            env["VOYAGE_API_KEY"] = api_key
            if log:
                log.info("User entered Voyage API key (value redacted)")

    elif provider == "ollama":
        # Clean up Voyage config if switching from Voyage
        if "VOYAGE_API_KEY" in env:
            del env["VOYAGE_API_KEY"]
            if log:
                log.info("Removed stale VOYAGE_API_KEY from env (switched to Ollama)")

        print()
        if not setup_ollama():
            print("\n  ERROR: Ollama setup failed.")
            sys.exit(1)
        env["OLLAMA_MODEL"] = OLLAMA_MODEL

    save_env(env)

    # =========================================================================
    # Step 5: Install Database
    # =========================================================================
    print_step(5, total_steps, "Install Database")
    log_section(log, "Step 5: Install Database")

    print("  The RAG database contains pre-indexed embeddings for fast search.")
    print("  Choose how you want to set up the database:")
    print()

    data_dir = SCRIPT_DIR / "data"
    provider_dir = data_dir / provider
    lancedb_dir = provider_dir / "lancedb"

    db_options = [
        ("Use Local Database",
         "Point to an existing LanceDB folder on disk."),
        ("Build Locally",
         "Run the ingest pipeline to build the database from scratch.\n"
         "      This may take 15-30 minutes depending on your hardware."),
    ]

    db_choice = prompt_choice(db_options, "Select database source")

    if db_choice == 0:
        # ---- Use Local Database ----
        print()
        print("  Select an existing LanceDB folder.")
        print("  It should contain .lance table directories")
        print(f"  (e.g. {', '.join(DATA_TABLES)}).")
        print()

        local_db_path = None
        folder = open_folder_picker("Select LanceDB folder")
        if folder:
            local_db_path = folder
        else:
            local_db_path = input("  Enter path to LanceDB folder: ").strip()

        if not local_db_path:
            print("  ERROR: No folder provided.")
            sys.exit(1)

        local_db = Path(local_db_path)
        if not local_db.exists():
            print(f"  ERROR: Folder does not exist: {local_db}")
            sys.exit(1)

        # Validate .lance table directories
        missing_tables = [t for t in DATA_TABLES if not (local_db / t).exists()]
        if missing_tables:
            print(f"  WARNING: Missing tables: {', '.join(missing_tables)}")
            if not prompt_yes_no("Continue anyway?", default=False):
                sys.exit(1)

        # Verify the database files
        print()
        success, error = verify_database_files(local_db)
        if not success:
            print(f"  WARNING: Verification issue: {error}")
            if not prompt_yes_no("Continue anyway?", default=False):
                sys.exit(1)

        # Symlink or copy into the expected provider data directory
        provider_dir.mkdir(parents=True, exist_ok=True)
        if lancedb_dir.exists() or lancedb_dir.is_symlink():
            shutil.rmtree(lancedb_dir, ignore_errors=True)
            if lancedb_dir.is_symlink():
                lancedb_dir.unlink()

        try:
            # Try symlink first (avoids duplicating data)
            lancedb_dir.symlink_to(local_db, target_is_directory=True)
            print(f"  Linked database: {local_db} -> {lancedb_dir}")
        except OSError:
            # Symlink may fail on Windows without privileges; fall back to copy
            print("  Symlink not available, copying database...", flush=True)
            shutil.copytree(local_db, lancedb_dir)
            print(f"  Copied database to {lancedb_dir}")

        if log:
            log.info(f"Local database path: {local_db}")
        print("  Local database configured successfully!")

    elif db_choice == 1:
        # ---- Build Locally ----
        print()
        print("  Building database locally using the ingest pipeline...")
        print("  This may take 15-30 minutes depending on your hardware.")
        print()

        # Ensure npm deps are installed first
        if not setup_npm():
            print("  ERROR: npm is required to build the database locally.")
            sys.exit(1)
        print()
        print("  Installing npm dependencies...", flush=True)
        exit_code, output = run_command(["npm", "install"], cwd=SCRIPT_DIR)
        if exit_code != 0:
            print(f"  WARNING: npm install had issues: {output[:200]}")
        else:
            print("  Dependencies installed.")
        print()

        # Derive paths for ingest scripts
        install_path_obj = Path(env["HYTALE_INSTALL_PATH"])
        decompiled_dir = Path(env.get("HYTALE_DECOMPILED_DIR", str(DECOMPILED_DIR)))
        client_data = install_path_obj / "Client" / "Data"
        assets_zip = install_path_obj / "Assets.zip"

        # Build ingest env vars
        ingest_env = {"EMBEDDING_PROVIDER": provider}
        if provider == "voyage":
            ingest_env["VOYAGE_API_KEY"] = env.get("VOYAGE_API_KEY", "")
        elif provider == "ollama":
            ingest_env["OLLAMA_MODEL"] = env.get("OLLAMA_MODEL", "nomic-embed-text")

        sources = [("Server Code", "src/ingest.ts", [str(decompiled_dir)])]
        if client_data.exists():
            sources.append(("Client UI", "src/ingest-client.ts", [str(client_data)]))
        if assets_zip.exists():
            sources.append(("Game Data", "src/ingest-gamedata.ts", [str(assets_zip)]))
        sources.append(("Docs", "src/ingest-docs.ts", []))

        build_failed = False
        for name, script, args in sources:
            print(f"  [{provider}/{name}] Indexing...")
            print()
            cmd = ["npx", "--yes", "tsx", script] + args
            exit_code, output = run_command(cmd, cwd=SCRIPT_DIR, env=ingest_env)
            if exit_code != 0:
                print(f"  [{provider}/{name}] Failed!")
                build_failed = True
            else:
                print(f"  [{provider}/{name}] Complete!")
            print()

        if build_failed:
            print("  WARNING: Some ingest steps failed.")
            if not prompt_yes_no("Continue anyway?", default=False):
                sys.exit(1)
        else:
            print("  Database built successfully!")

    # ---- Custom Docs (applies to all modes) ----
    print()
    if prompt_yes_no("Ingest docs from a custom folder?", default=False):
        print()
        print("  Select a folder containing .md or .html documentation files.")
        print()

        custom_docs_path = None
        folder = open_folder_picker("Select Custom Docs folder")
        if folder:
            custom_docs_path = folder
        else:
            custom_docs_path = input("  Enter path to custom docs folder: ").strip()

        if custom_docs_path:
            custom_dir = Path(custom_docs_path)
            if not custom_dir.exists():
                print(f"  WARNING: Folder does not exist: {custom_dir}")
            elif not any(custom_dir.rglob("*.md")) and not any(custom_dir.rglob("*.html")):
                print("  WARNING: No .md or .html files found in folder.")
                if prompt_yes_no("Continue with this folder anyway?", default=False):
                    env["CUSTOM_DOCS_DIR"] = str(custom_dir)
                    save_env(env)
            else:
                env["CUSTOM_DOCS_DIR"] = str(custom_dir)
                save_env(env)

                # Run docs ingest with custom path
                print()
                print("  Ingesting custom docs...", flush=True)
                ingest_env = {"EMBEDDING_PROVIDER": provider}
                if provider == "voyage":
                    ingest_env["VOYAGE_API_KEY"] = env.get("VOYAGE_API_KEY", "")
                elif provider == "ollama":
                    ingest_env["OLLAMA_MODEL"] = env.get("OLLAMA_MODEL", "nomic-embed-text")

                exit_code, output = run_command(
                    ["npx", "--yes", "tsx", "src/ingest-docs.ts", str(custom_dir)],
                    cwd=SCRIPT_DIR,
                    env=ingest_env,
                )
                if exit_code != 0:
                    print("  WARNING: Custom docs ingest failed.")
                else:
                    print("  Custom docs ingested successfully!")
        else:
            print("  Skipped: No folder provided.")

    # Ensure npm is installed
    print()
    if not setup_npm():
        print()
        print("  WARNING: npm is required for the MCP server.")
        print("  You can continue setup, but MCP integration won't work until npm is installed.")
        if not prompt_yes_no("Continue anyway?", default=False):
            sys.exit(1)
    else:
        # Install npm dependencies
        print()
        print("  Installing npm dependencies...", flush=True)
        exit_code, output = run_command(["npm", "install"], cwd=SCRIPT_DIR)
        if exit_code != 0:
            print(f"  WARNING: npm install had issues: {output[:200]}")
        else:
            print("  Dependencies installed.")

    # =========================================================================
    # Step 6: AI Tool Integration
    # =========================================================================
    print_step(6, total_steps, "AI Tool Integration (MCP)")
    log_section(log, "Step 6: AI Tool Integration (MCP)")

    print("  The RAG server uses the Model Context Protocol (MCP) to integrate")
    print("  with AI coding assistants. Select which tools you want to configure:")
    print()

    # Build options list from MCP_CLIENTS
    client_ids = list(MCP_CLIENTS.keys())
    client_options = [
        (MCP_CLIENTS[cid]["name"], MCP_CLIENTS[cid]["description"])
        for cid in client_ids
    ]

    selected_indices = prompt_multi_choice(client_options, "Select AI tools to configure")

    if selected_indices:
        selected_clients = [client_ids[i] for i in selected_indices]
        log.info(f"Selected MCP clients: {selected_clients}")
        print()
        print(f"  Configuring {len(selected_clients)} tool(s)...")
        print()

        results = setup_mcp_clients(selected_clients, SCRIPT_DIR)
        log.info(f"MCP setup results: {results}")

        # Show summary
        successful = [c for c, success in results.items() if success]
        failed = [c for c, success in results.items() if not success]

        print()
        print("  +----------------------------------------------------------+")
        print("  |                   Setup Complete!                        |")
        print("  +----------------------------------------------------------+")
        print()

        if successful:
            print("  MCP integration configured for:")
            for client_id in successful:
                print(f"    - {MCP_CLIENTS[client_id]['name']}")
            print()
            print("  To use the Hytale RAG:")
            print("    1. Restart your AI tool (or open a new session)")
            print("    2. Ask it to search the Hytale codebase, e.g.:")
            print("       - 'Search the Hytale code for player movement'")
            print("       - 'Find methods related to inventory'")
            print("       - 'How does the NPC AI work?'")

        if failed:
            print()
            print("  Failed to configure:")
            for client_id in failed:
                print(f"    - {MCP_CLIENTS[client_id]['name']}")
    else:
        log.info("No MCP clients selected")
        print()
        print("  +----------------------------------------------------------+")
        print("  |                   Setup Complete!                        |")
        print("  +----------------------------------------------------------+")
        print()
        print("  No AI tool integration configured.")
        print()
        print("  To use the RAG server manually:")
        print()
        print(f"    cd {SCRIPT_DIR}")
        print("    npx tsx src/search.ts \"your search query\"")
        print()
        print("  Examples:")
        print("    npx tsx src/search.ts \"player movement\"")
        print("    npx tsx src/search.ts \"inventory\" --limit 10")
        print("    npx tsx src/search.ts --stats")
        print()
        print("  Run setup.py again to configure AI tool integration later.")

    print()
    print("  ----------------------------------------------------------")
    print("  Additional tools:")
    print("    - index-all.py  : Re-index the database after Hytale updates")
    if DECOMPILED_DIR.exists():
        print(f"    - Decompiled source: {DECOMPILED_DIR}")
    if JAVADOCS_DIR.exists():
        print(f"    - Javadocs: {JAVADOCS_DIR / 'index.html'}")
    print("  ----------------------------------------------------------")
    print()

    # Show log file location
    if log_file:
        print(f"  Log file: {log_file}")
        print("  (Share this file if you need help troubleshooting)")
        print()
        log.info("=== Setup completed successfully ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if log:
            log.warning("Setup cancelled by user (Ctrl+C)")
        print("\n\n  Setup cancelled by user.")
        if log_file:
            print(f"  Log file: {log_file}")
        sys.exit(1)
    except Exception as e:
        if log:
            log.error(f"Unhandled exception: {e}")
            log_exception(log, "main")
        print(f"\n\n  ERROR: An unexpected error occurred: {e}")
        if log_file:
            print(f"  Log file: {log_file}")
            print("  Please share this log file when reporting the issue.")
        sys.exit(1)
