# method_exporter.py
import re
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

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
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = (Path(__file__).resolve().parent / self.config_path).resolve()
        self.obsidian_base: Optional[Path] = None
        self.components: Dict = {}
        self.target_files: List[Dict] = []
        self.max_file_size: int = 0
        self.ignore_dirs: List[Path] = []  # resolved absolute paths
        self.ignore_files: List[Path] = []  # resolved absolute paths
        self.ignore_dir_names: set[str] = set()
        self.load_config()

        self.timestamp = datetime.now().strftime("%H-%M-%S_%m-%d-%y")
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

        export_base_value = config.get("export_base", "")
        if not isinstance(export_base_value, str) or not export_base_value.strip():
            print("Config missing required key: 'export_base'. Stopping.")
            sys.exit(1)
        self.obsidian_base = Path(export_base_value)

        components_value = config.get("components", {})
        if not isinstance(components_value, dict):
            print("Config key 'components' must be an object. Stopping.")
            sys.exit(1)
        self.components = components_value

        target_files_value = config.get("target_files", [])
        if not isinstance(target_files_value, list):
            print("Config key 'target_files' must be a list. Stopping.")
            sys.exit(1)
        self.target_files = target_files_value

        max_file_size_value = config.get("max_file_size_mb", 10)
        if not isinstance(max_file_size_value, (int, float)):
            print("Config key 'max_file_size_mb' must be numeric. Stopping.")
            sys.exit(1)
        self.max_file_size = int(max_file_size_value * 1024 * 1024)

        # Resolve ignore lists to absolute Paths for reliable comparison.
        # ignore_dir_names matches any path segment by name (e.g. "__pycache__").
        # ignore_dirs matches full absolute paths to specific directories.
        # ignore_files matches full absolute paths to specific files.
        self.ignore_dirs = [Path(d).resolve() for d in config.get("ignore_dirs", [])]
        self.ignore_dir_names = set(config.get("ignore_dir_names", []))
        self.ignore_files = [Path(f).resolve() for f in config.get("ignore_files", [])]
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
    #   method_<method_name>_<path>   named method from <path>
    #                                 path boundary is detected from the
    #                                 first real path start marker
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

    def _make_target_entry(
            self,
            file_path: Path,
            mode: str,
            *,
            comment: str = "",
            methods: Optional[List[str]] = None,
            origin_token: str = "",
            source_kind: str = "",
            source_path: str = "",
            recursive: bool = False,
            extensions: Optional[List[str]] = None,
    ) -> Dict:
        """
        Build a target_files entry while preserving origin metadata for export naming.

        The leading-underscore metadata fields are internal-only and do not affect
        extraction behavior. They exist so export_all() can infer better filenames
        from the actual target selection and token flags.
        """
        entry = {
            "file": str(file_path),
            "mode": mode,
            "comment": comment,
            "_origin_token": origin_token,
            "_origin_component": None,
            "_source_kind": source_kind,
            "_source_path": source_path,
            "_recursive": recursive,
            "_extensions": list(extensions or []),
        }
        if methods is not None:
            entry["methods"] = list(methods)
        return entry

    def _token_file(self, token: str) -> list:
        path_str = token[len("file_"):]
        p = Path(path_str)
        if not p.exists():
            print(f"[WARN] File not found, skipping: {p}")
            return []
        if self._is_ignored(p):
            print(f"[IGNORE] Skipping: {p}")
            return []
        return [self._make_target_entry(
            p,
            "file",
            comment="",
            origin_token=token,
            source_kind="file",
            source_path=str(p),
            recursive=False,
            extensions=[p.suffix.lower()] if p.suffix else []
        )]

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

        normalized_exts = []
        if extensions:
            normalized_exts = sorted(
                {e if e.startswith(".") else f".{e}" for e in extensions}
            )

        return [
            self._make_target_entry(
                f,
                "file",
                comment="",
                origin_token=token,
                source_kind="dir",
                source_path=str(base),
                recursive=recursive,
                extensions=normalized_exts
            )
            for f in files
        ]

    def _split_method_token_body(self, body: str) -> tuple[Optional[str], Optional[str]]:
        """
        Parse:
            method_<method_name>_<path>

        Priority:
          1. Explicit path-start markers
          2. First underscore whose remainder resolves to an existing path
          3. Final legacy fallback
        """
        candidates = []

        for marker in ("_/", "_./", "_../", "_~/"):
            idx = body.find(marker)
            if idx != -1:
                candidates.append(idx)

        win_match = re.search(r'_(?=[A-Za-z]:[\\/])', body)
        if win_match:
            candidates.append(win_match.start())

        if candidates:
            sep = min(candidates)
            method_name = body[:sep]
            path_str = body[sep + 1:]
            if method_name and path_str:
                return method_name, path_str

        # Plain relative-path fallback:
        # choose the first underscore whose remainder is an actual existing path.
        for m in re.finditer(r"_", body):
            sep = m.start()
            method_name = body[:sep]
            path_str = body[sep + 1:]
            if not method_name or not path_str:
                continue
            if Path(path_str).exists():
                return method_name, path_str

        # Legacy final fallback
        sep = body.rfind("_")
        if sep == -1:
            return None, None

        method_name = body[:sep]
        path_str = body[sep + 1:]

        if not method_name or not path_str:
            return None, None

        return method_name, path_str

    def _token_method(self, token: str) -> list:
        """
        method_<method_name>_<path>

        Uses explicit path-start detection so underscores inside Linux/Windows
        paths do not corrupt parsing.
        """
        body = token[len("method_"):]
        method_name, path_str = self._split_method_token_body(body)

        if method_name is None or path_str is None:
            print(f"[WARN] method_ token missing path separator, skipping: '{token}'")
            return []

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

        return [self._make_target_entry(
            p,
            "methods",
            methods=[method_name],
            comment="",
            origin_token=token,
            source_kind="method",
            source_path=str(p),
            recursive=False,
            extensions=[p.suffix.lower()] if p.suffix else []
        )]

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
            expanded = self.parse_token(entry)
            for item in expanded:
                item["_origin_component"] = name
            results.extend(expanded)
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

    @staticmethod
    def _build_file_tree(paths: list) -> str:
        """Build a directory tree of only the exported file paths."""
        if not paths:
            return ""

        split = [Path(p).parts for p in paths]

        # Find common root depth
        common_len = 0
        for level in zip(*split):
            if len(set(level)) == 1:
                common_len += 1
            else:
                break

        root_label = str(Path(*split[0][:common_len])) if common_len else "/"

        # Build nested dict tree
        tree: dict = {}
        for parts in split:
            node = tree
            for part in parts[common_len:]:
                node = node.setdefault(part, {})

        # Render with tree characters
        lines = [root_label]

        def _render(node: dict, prefix: str = "") -> None:
            items = sorted(node.keys())
            for i, key in enumerate(items):
                is_last = i == len(items) - 1
                lines.append(f"{prefix}{'└── ' if is_last else '├── '}{key}")
                if node[key]:
                    _render(node[key], prefix + ("    " if is_last else "│   "))

        _render(tree)
        return "\n".join(lines)

    def extract_all_methods(self):
        self.extracted_methods = []
        for file_config in self.target_files:
            if not isinstance(file_config, dict):
                print(f"Error: Invalid target entry (expected dict): {file_config}")
                continue

            source_file_value = file_config.get("file")
            if not isinstance(source_file_value, str) or not source_file_value.strip():
                print(f"Error: Invalid target entry missing file path: {file_config}")
                continue

            source_path = Path(source_file_value)
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

    @staticmethod
    def _slugify_filename_part(value: str) -> str:
        """
        Keep filenames readable and filesystem-safe while preserving underscores.
        """
        value = (value or "").strip()
        if not value:
            return "METHOD_SANDBOX"
        value = re.sub(r"[^\w\-. ]+", "", value)
        value = re.sub(r"\s+", "_", value)
        value = re.sub(r"_+", "_", value)
        value = value.strip("._")
        return value or "METHOD_SANDBOX"

    def _infer_export_name(self) -> str:
        """
        Best-guess export stem based on the actual selected targets and token flags.

        Priority:
          1. Single directory-origin export:
             <dirname>_full_export[_recursive][_<ext>...]
          2. Single file target:
             <file_stem>_full_file_export
          3. Single method target:
             <file_stem>_<method>_method_export
          4. Single component fallback:
             <component>_export
          5. Common root fallback:
             <rootname>_export
          6. METHOD_SANDBOX
        """
        if not self.target_files:
            return "METHOD_SANDBOX"

        # Directory-origin inference (works for direct dir_... tokens and components
        # that expand from a single dir_... entry)
        dir_items = [
            item for item in self.target_files
            if item.get("_source_kind") == "dir" and item.get("_source_path")
        ]
        unique_dir_sources = list(dict.fromkeys(
            item["_source_path"] for item in dir_items
        ))

        if len(unique_dir_sources) == 1:
            source_dir = Path(unique_dir_sources[0])
            parts = [source_dir.name or "export", "full_export"]

            if any(bool(item.get("_recursive")) for item in dir_items):
                parts.append("recursive")

            ext_parts = []
            for item in dir_items:
                for ext in item.get("_extensions", []):
                    cleaned = str(ext).lstrip(".").lower()
                    if cleaned and cleaned not in ext_parts:
                        ext_parts.append(cleaned)

            if ext_parts:
                parts.extend(ext_parts)

            return self._slugify_filename_part("_".join(parts))

        # Single target fallback
        if len(self.target_files) == 1:
            item = self.target_files[0]
            source_file = Path(item["file"])

            if item.get("mode") == "file":
                return self._slugify_filename_part(
                    f"{source_file.stem}_full_file_export"
                )

            if item.get("mode") == "methods":
                methods = item.get("methods", [])
                if len(methods) == 1:
                    return self._slugify_filename_part(
                        f"{source_file.stem}_{methods[0]}_method_export"
                    )
                return self._slugify_filename_part(
                    f"{source_file.stem}_method_export"
                )

        # Single component fallback
        components = [
            item.get("_origin_component")
            for item in self.target_files
            if item.get("_origin_component")
        ]
        unique_components = list(dict.fromkeys(components))
        if len(unique_components) == 1:
            return self._slugify_filename_part(f"{unique_components[0]}_export")

        # Common-root fallback from concrete exported files
        try:
            exported_paths = [Path(item["file"]) for item in self.target_files if item.get("file")]
            if exported_paths:
                split = [p.parts for p in exported_paths]
                common_len = 0
                for level in zip(*split):
                    if len(set(level)) == 1:
                        common_len += 1
                    else:
                        break
                if common_len:
                    common_root = Path(*split[0][:common_len])
                    if common_root.name:
                        return self._slugify_filename_part(f"{common_root.name}_export")
        except Exception:
            pass

        return "METHOD_SANDBOX"

    def export_all(self):
        if not self.extracted_methods:
            print("No items to export.")
            return

        if self.obsidian_base is None:
            print("Error: export base not configured.")
            return

        output_stem = self._infer_export_name()
        output_filename = f"{output_stem}_{self.timestamp}.md"
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

                # ── INSERT STARTS ──
                exported_paths = list(dict.fromkeys(
                    item["source_file"] for item in self.extracted_methods
                ))
                if exported_paths:
                    f.write(f"```\n{self._build_file_tree(exported_paths)}\n```\n\n")
                # ── INSERT ENDS ──

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
    config_path = Path(__file__).resolve().with_name("method_export_config.json")

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
        with open(config_path, "w", encoding="utf-8") as f:
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