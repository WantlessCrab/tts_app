# method_exporter.py
import re
import json
import random
import string
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Reference set for common source extensions.
# Not automatically applied — only used when explicitly requested via:
#     dir_/path:.py,.yml    --or--
#     dir_/path:r:.py
DEFAULT_EXTENSIONS = {
    ".py", ".js", ".yml", ".yaml", ".json",
    ".txt", ".html", ".toml", ".sh", ".rs"
}


class MethodExtractor:
    def __init__(self, config_path="method_export_config.json"):
        self.config_path = config_path
        self.obsidian_base: Optional[Path] = None
        self.components: Dict = {}
        self.target_files: List[Dict] = []
        self.max_file_size: int = 0
        self.ignore_dirs: List[Path] = []  # resolved absolute paths
        self.ignore_files: List[Path] = []  # resolved absolute paths
        self.load_config()

        self.timestamp = datetime.now().strftime("%I-%M-%S_%d-%m-%y") + "_" + "".join(
            random.choices(string.ascii_lowercase + string.digits, k=8)
        )
        self.extracted_methods: List[Dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIG
    # ─────────────────────────────────────────────────────────────────────────

    def load_config(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Stopping.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Config file {self.config_path} is not valid JSON. Stopping.")
            sys.exit(1)

        self.obsidian_base = Path(config.get("export_base", ""))
        self.components = config.get("components", {})
        self.target_files = config.get("target_files", [])
        self.max_file_size = config.get("max_file_size_mb", 10) * 1024 * 1024

        # Resolve ignore lists to absolute Paths for reliable comparison.
        # ignore_dir_names matches any path segment by name (e.g. "__pycache__").
        # ignore_dirs matches full absolute paths to specific directories.
        # ignore_files matches full absolute paths to specific files.
        self.ignore_dirs = [Path(d).resolve() for d in config.get("ignore_dirs", [])]
        self.ignore_dir_names = set(config.get("ignore_dir_names", []))
        self.ignore_files = [Path(f).resolve() for f in config.get("ignore_files", [])]

        if not str(self.obsidian_base):
            print("Config missing required key: 'export_base'. Stopping.")
            sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # TOKEN PARSER
    # ─────────────────────────────────────────────────────────────────────────
    #
    # TOKEN PROTOCOL
    # Identical syntax for CLI args AND config component entries.
    # parse_token handles both — zero duplicated logic.
    # ─────────────────────────────────────────────────────────────────────────
    #   file_<path>                   full-file entry
    #
    #   dir_<path>                    all files directly inside <path>  (flat)
    #   dir_<path>:<ext>,<ext>        flat, filtered by extension(s)
    #   dir_<path>:r                  recursive, all files
    #   dir_<path>:r:<ext>,<ext>      recursive, filtered
    #
    #   method_<n>_<path>          named method from <path>
    #                                 splits on LAST underscore —
    #                                 snake_case names fully safe
    #
    #   component_<n>              expand named group from config["components"]
    #
    # ─────────────────────────────────────────────────────────────────────────
    # EXAMPLES
    # ─────────────────────────────────────────────────────────────────────────
    #   python method_exporter.py component_processor
    #   python method_exporter.py component_gateway component_processor
    #   python method_exporter.py file_/home/wantless/TTS/my_app/audio_server.py
    #   python method_exporter.py dir_/home/wantless/TTS:.yml,.txt
    #   python method_exporter.py dir_/home/wantless/TTS/my_app:r:.py
    #   python method_exporter.py method_compile_tts_ready_content_/home/wantless/TTS/my_app/pdf_processor/extraction_engine.py
    #   python method_exporter.py component_processor file_/home/wantless/TTS/my_app/gateway_router.py

    def parse_token(self, token: str) -> list:
        """Single dispatcher — used for both CLI args and config component entries."""
        if token.startswith("file_"):
            return self._token_file(token)
        if token.startswith("dir_"):
            return self._token_dir(token)
        if token.startswith("method_"):
            return self._token_method(token)
        if token.startswith("component_"):
            return self._token_component(token[len("component_"):])
        print(f"[WARN] Unrecognised token, skipping: '{token}'")
        return []

    def _is_ignored(self, p: Path) -> bool:
        """Returns True if path matches any ignore rule."""
        resolved = p.resolve()
        # Exact file match
        if resolved in self.ignore_files:
            return True
        # Exact directory match or subdirectory of ignored dir
        for ignored_dir in self.ignore_dirs:
            try:
                resolved.relative_to(ignored_dir)
                return True
            except ValueError:
                pass
        # Any path segment matches an ignored dir name
        if self.ignore_dir_names.intersection(set(resolved.parts)):
            return True
        return False

    def _token_file(self, token: str) -> list:
        path_str = token[len("file_"):]
        p = Path(path_str)
        if not p.exists():
            print(f"[WARN] File not found, skipping: {p}")
            return []
        if self._is_ignored(p):
            print(f"[IGNORE] Skipping: {p}")
            return []
        return [{"file": str(p), "mode": "file", "comment": ""}]

    def _token_dir(self, token: str) -> list:
        """
        dir_<path>                  flat, all files
        dir_<path>:<ext>,<ext>      flat, filtered
        dir_<path>:r                recursive, all files
        dir_<path>:r:<ext>,<ext>    recursive, filtered
        """
        body = token[len("dir_"):]
        parts = body.split(":", 2)
        dir_str = parts[0]
        recursive = False
        extensions = []

        if len(parts) >= 2:
            if parts[1] == "r":
                recursive = True
                if len(parts) == 3:
                    extensions = [e.strip() for e in parts[2].split(",") if e.strip()]
            else:
                extensions = [e.strip() for e in parts[1].split(",") if e.strip()]

        base = Path(dir_str)
        if not base.exists():
            print(f"[WARN] Directory not found, skipping: {base}")
            return []

        walker = base.rglob("*") if recursive else base.glob("*")
        files = sorted(f for f in walker if f.is_file())

        if extensions:
            ext_set = {e if e.startswith(".") else f".{e}" for e in extensions}
            files = [f for f in files if f.suffix.lower() in ext_set]

        # Apply ignore rules
        files = [f for f in files if not self._is_ignored(f)]

        if not files:
            print(f"[WARN] No matching files in directory: {base}")
            return []

        return [{"file": str(f), "mode": "file", "comment": ""} for f in files]

    def _token_method(self, token: str) -> list:
        """
        method_<n>_<path>
        Splits on LAST underscore. Snake_case method names are fully safe.
        """
        body = token[len("method_"):]
        last_sep = body.rfind("_")
        if last_sep == -1:
            print(f"[WARN] method_ token missing path separator, skipping: '{token}'")
            return []
        method_name = body[:last_sep]
        path_str = body[last_sep + 1:]
        if not method_name:
            print(f"[WARN] method_ token has empty method name, skipping: '{token}'")
            return []
        p = Path(path_str)
        if not p.exists():
            print(f"[WARN] File not found for method entry, skipping: {p}")
            return []

        if self._is_ignored(p):
            print(f"[IGNORE] Skipping: {p}")
            return []

        return [{"file": str(p), "mode": "methods", "methods": [method_name], "comment": ""}]

    def _token_component(self, name: str) -> list:
        """
        Expands a named component group.
        Component entries in config["components"] are token strings —
        same syntax as CLI args. parse_token handles both.
        """
        if name not in self.components:
            print(f"[WARN] Component '{name}' not found in config, skipping.")
            return []
        entries = self.components[name]
        if not isinstance(entries, list):
            print(f"[WARN] Component '{name}' value is not a list, skipping.")
            return []
        results = []
        for entry in entries:
            results.extend(self.parse_token(entry))
        return results

    @staticmethod
    def _deduplicate(target_files: list) -> list:
        """
        Removes exact duplicate (file, mode) pairs.
        For mode='methods', merges method lists rather than dropping.
        """
        seen = {}
        output = []
        for entry in target_files:
            key = (entry.get("file"), entry.get("mode"))
            if key not in seen:
                seen[key] = len(output)
                output.append(dict(entry))
            else:
                if entry.get("mode") == "methods":
                    existing = output[seen[key]]
                    merged = list(dict.fromkeys(
                        existing.get("methods", []) + entry.get("methods", [])
                    ))
                    existing["methods"] = merged
        return output

    # ─────────────────────────────────────────────────────────────────────────
    # METHOD EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def extract_method(self, content: str, method_name: str) -> Optional[Dict]:
        patterns = [
            rf'^\s*def\s+{re.escape(method_name)}\s*\([^)]*\)[^:]*:',
            rf'^\s*async\s+def\s+{re.escape(method_name)}\s*\([^)]*\)[^:]*:',
            rf'^\s*{re.escape(method_name)}\s*\([^)]*\)\s*\{{',
            rf'^\s*{re.escape(method_name)}\s*=\s*(?:async\s*)?\([^)]*\)\s*=>\s*\{{',
            rf'^\s*async\s+{re.escape(method_name)}\s*\([^)]*\)\s*\{{'
        ]
        for pattern in patterns:
            regex = re.compile(pattern, re.MULTILINE)
            match = regex.search(content)
            if match:
                start_pos = match.start()
                is_python = (
                        pattern.startswith(r'^\s*def') or
                        pattern.startswith(r'^\s*async\s+def')
                )
                end_pos = (
                    self._find_python_method_block(content, start_pos)
                    if is_python
                    else self._find_js_method_end(content, start_pos)
                )
                method_content = content[start_pos:end_pos]
                line_number = content[:start_pos].count('\n') + 1
                return {
                    "name": method_name,
                    "content": method_content,
                    "line_number": line_number,
                    "char_count": len(method_content)
                }
        return None

    def _find_js_method_end(self, content: str, start_pos: int) -> int:
        sig_end_pos = content.find(")", start_pos)
        if sig_end_pos == -1:
            return len(content)
        start_brace_pos = -1
        i = sig_end_pos + 1
        while i < len(content):
            char = content[i]
            if char == ";":
                return i + 1
            if char == "{":
                start_brace_pos = i
                break
            i += 1
        if start_brace_pos == -1:
            return len(content)
        brace_count = 1
        current_pos = start_brace_pos + 1
        while current_pos < len(content):
            char = content[current_pos]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return current_pos + 1
            current_pos += 1
        return len(content)

    def _find_python_method_block(self, content: str, start_pos: int) -> int:
        lines = content.split("\n")
        start_line_idx = content[:start_pos].count("\n")
        total_lines = len(lines)
        actual_def_line = None
        for i in range(start_line_idx, min(start_line_idx + 10, total_lines)):
            stripped = lines[i].lstrip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                actual_def_line = i
                break
        if actual_def_line is None:
            return len(content)
        def_line = lines[actual_def_line]
        base_indent = len(def_line) - len(def_line.lstrip(" \t"))
        for i in range(actual_def_line + 1, total_lines):
            line = lines[i]
            stripped = line.lstrip()
            if not stripped:
                continue
            indent = len(line) - len(line.lstrip(" \t"))
            if indent <= base_indent:
                if (
                        stripped.startswith("def ") or
                        stripped.startswith("async def ") or
                        stripped.startswith("class ") or
                        stripped.startswith("@")
                ):
                    return sum(len(lines[j]) + 1 for j in range(i))
        return len(content)

    # ─────────────────────────────────────────────────────────────────────────
    # EXTRACTION DRIVER
    # ─────────────────────────────────────────────────────────────────────────

    def extract_all_methods(self):
        self.extracted_methods = []
        for file_config in self.target_files:
            source_path = Path(file_config["file"])
            mode = file_config.get("mode", "methods")

            if not source_path.exists():
                print(f"Error: Target file not found: {source_path}")
                continue

            try:
                if source_path.stat().st_size > self.max_file_size:
                    print(f"Error: File too large: {source_path}")
                    continue
            except (OSError, PermissionError) as e:
                print(f"Cannot access {source_path}: {e}")
                continue

            try:
                with open(source_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {source_path}: {e}")
                continue

            print(f"\nProcessing: {source_path}  [{mode}]")

            if mode == "file":
                self.extracted_methods.append({
                    "name": source_path.name,
                    "content": content,
                    "line_number": 1,
                    "char_count": len(content),
                    "source_file": str(source_path),
                    "is_full_file": True
                })
                print(f"  ✓ {len(content):,} chars")

            elif mode == "methods":
                for method_name in file_config.get("methods", []):
                    print(f"  Extracting: {method_name}...", end=" ")
                    method_data = self.extract_method(content, method_name)
                    if method_data:
                        method_data["source_file"] = str(source_path)
                        method_data["is_full_file"] = False
                        self.extracted_methods.append(method_data)
                        print(
                            f"✓ (line {method_data['line_number']}, {method_data['char_count']} chars)")
                    else:
                        print("✗ NOT FOUND")
            else:
                print(f"  [WARN] Unknown mode '{mode}' - skipping")

        print(f"\n{'=' * 60}")
        print(f"Total items extracted: {len(self.extracted_methods)}")
        print(f"{'=' * 60}")

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    def export_all(self):
        if not self.extracted_methods:
            print("No items to export.")
            return

        output_filename = f"METHOD_SANDBOX_{self.timestamp}.md"
        output_path = self.obsidian_base / output_filename

        try:
            self.obsidian_base.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# Method Extraction Sandbox\n\n")
                f.write(f"**Extracted**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Total Items**: {len(self.extracted_methods)}\n")
                f.write(f"**Source Files**: {len(self.target_files)}\n\n")
                f.write("---\n\n")

                current_file = None
                for item in self.extracted_methods:
                    if item["source_file"] != current_file:
                        current_file = item["source_file"]
                        f.write(f"## Source: `{current_file}`\n\n")

                    if item.get("is_full_file", False):
                        f.write(f"### Full File: `{item['name']}`\n\n")
                        f.write(f"**Size**: {item['char_count']:,} characters\n\n")
                    else:
                        f.write(f"### Method: `{item['name']}()`\n\n")
                        f.write(f"**Line Number**: {item['line_number']}\n")
                        f.write(f"**Size**: {item['char_count']:,} characters\n\n")

                    ext = Path(current_file).suffix.lower()
                    lang = {
                        ".py": "python", ".js": "javascript",
                        ".yml": "yaml", ".yaml": "yaml",
                        ".json": "json", ".html": "html",
                        ".toml": "toml", ".sh": "bash", ".rs": "rust"
                    }.get(ext, "text")

                    f.write(f"```{lang}\n")
                    f.write(item["content"])
                    f.write("\n```\n\n---\n\n")

            print(f"\n{'=' * 60}")
            print(f"✓ Export complete: {len(self.extracted_methods)} items")
            print(f"✓ Output: {output_path}")
            print(f"{'=' * 60}")

        except Exception as e:
            print(f"Error writing output: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config_path = "method_export_config.json"

    if not Path(config_path).exists():
        sample = {
            "export_base": "C:/Users/wantless/Documents/Obsidian Vault/code_exports",
            "components": {
                "example_service": [
                    "file_./my_app/audio_server.py"
                ]
            },
            "target_files": [],
            "max_file_size_mb": 10
        }
        with open(config_path, "w") as f:
            json.dump(sample, f, indent=4)
        print(f"No config found. Created sample at: {config_path}")
        print("Edit export_base and run again.")
        sys.exit(0)

    extractor = MethodExtractor(config_path)

    cli_tokens = sys.argv[1:]
    if cli_tokens:
        cli_entries = []
        for token in cli_tokens:
            cli_entries.extend(extractor.parse_token(token))

        if cli_entries:
            # CLI entries append to config target_files baseline, then dedup.
            # No CLI args = config target_files used as-is (backward compatible).
            extractor.target_files = extractor._deduplicate(
                extractor.target_files + cli_entries
            )
            print(f"[CLI] {len(cli_entries)} entries injected "
                  f"({len(extractor.target_files)} total after dedup)")

    if not extractor.target_files:
        print("Nothing to export: no target_files in config and no valid CLI tokens.")
        sys.exit(0)

    extractor.extract_all_methods()
    extractor.export_all()