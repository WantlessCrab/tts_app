# method_exporter.py
import os
import re
import json
import random
import string
from pathlib import Path
from datetime import datetime
from typing import Set, Dict, List, Optional, Union

class MethodExtractor:
    def __init__(self, config_path="method_export_config.json"):
        self.config_path = config_path
        self.obsidian_base: Optional[Path] = None
        # ✅ CHANGE: Store array of file configs
        self.target_files: List[Dict] = []
        self.max_file_size: int = 0
        self.load_config()

        self.timestamp = datetime.now().strftime("%I-%M-%S_%d-%m-%y") + "_" + ''.join(
            random.choices(string.ascii_lowercase + string.digits, k=8))

        self.extracted_methods: List[Dict] = []

    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            self.obsidian_base = Path(config.get("export_base"))
            # ✅ CHANGE: Load array instead of single file
            self.target_files = config.get("target_files", [])
            self.max_file_size = config.get("max_file_size_mb", 10) * 1024 * 1024

            # ✅ CHANGE: Validate array not empty
            if not self.obsidian_base or not self.target_files:
                raise KeyError("Config missing 'export_base' or 'target_files'")

        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Stopping.")
            exit()
        except json.JSONDecodeError:
            print(f"Error: Config file {self.config_path} is not valid JSON. Stopping.")
            exit()
        except KeyError as e:
            print(f"Error: Config file missing required key: {e}. Stopping.")
            exit()

    def extract_method(self, content: str, method_name: str) -> Optional[Dict]:
        """Extract a complete method from JavaScript OR Python content"""

        patterns = [
            # Python function: Allow for return type hints (-> ...) before the colon
            # Changed: \s*:  -->  [^:]*:
            rf'^\s*def\s+{re.escape(method_name)}\s*\([^)]*\)[^:]*:',

            # Python async: Allow for return type hints
            # Changed: \s*:  -->  [^:]*:
            rf'^\s*async\s+def\s+{re.escape(method_name)}\s*\([^)]*\)[^:]*:',

            # JavaScript regular method (Unchanged)
            rf'^\s*{re.escape(method_name)}\s*\([^)]*\)\s*\{{',
            # JavaScript arrow function (Unchanged)
            rf'^\s*{re.escape(method_name)}\s*=\s*(?:async\s*)?\([^)]*\)\s*=>\s*\{{',
            # JavaScript async method (Unchanged)
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

                if is_python:
                    end_pos = self._find_python_method_block(content, start_pos)
                else:
                    end_pos = self._find_js_method_end(content, start_pos)

                method_content = content[start_pos:end_pos]
                line_number = content[:start_pos].count('\n') + 1

                return {
                    'name': method_name,
                    'content': method_content,
                    'line_number': line_number,
                    'char_count': len(method_content)
                }

        return None

    def _find_js_method_end(self, content: str, start_pos: int) -> int:
        """Find end of JS method by locating the starting brace and counting scope."""

        # 1. Find the end of the argument list (final closing parenthesis)
        # This is the point where the body *must* begin.
        sig_end_pos = content.find(')', start_pos)

        if sig_end_pos == -1:
            # Malformed signature or incomplete file - return full content
            return len(content)

        # 2. Search forward from the signature end for the method body's opening brace '{'
        start_brace_pos = -1
        i = sig_end_pos + 1
        while i < len(content):
            char = content[i]

            # Stop search if semicolon is found (Abstract/Non-bodied method case: e.g., 'throw new Error("...");')
            if char == ';':
                # End of statement is end of method
                return i + 1

            if char == '{':
                start_brace_pos = i
                break

            i += 1

        # --- MODAL FALLBACK ---
        if start_brace_pos == -1:
            # No opening brace found after signature (e.g., abstract method without semicolon, or arrow function with expression body)
            # We assume the body extends to the end of the file or next logical break.
            # Given the failure mode, returning len(content) is the safest structural fallback here.
            return len(content)

        # 3. Robust Brace Counting Logic (Used only if a body is present)
        brace_count = 1  # Start count at 1 for the opening brace we just found
        current_pos = start_brace_pos + 1

        while current_pos < len(content):
            char = content[current_pos]

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

                # 4. Method ends when the brace counter returns to zero
                if brace_count == 0:
                    # Return the position *after* the closing brace
                    return current_pos + 1

            current_pos += 1

        return len(content)

    def _find_python_method_block(self, content: str, start_pos: int) -> int:
        """Find Python method end by locating next top-level definition"""
        lines = content.split("\n")
        start_line_idx = content[:start_pos].count("\n")
        total_lines = len(lines)

        # CRITICAL FIX: Find the ACTUAL def line (might have decorators/blanks before start_pos)
        actual_def_line = None
        for i in range(start_line_idx, min(start_line_idx + 10, total_lines)):
            line = lines[i]
            stripped = line.lstrip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                actual_def_line = i
                break

        if actual_def_line is None:
            # Couldn't find def line - fallback
            print(f"    [WARNING] Could not find def line near {start_line_idx}")
            return len(content)

        # Get base indentation from the ACTUAL def line
        def_line = lines[actual_def_line]
        base_indent = len(def_line) - len(def_line.lstrip(" \t"))

        print(f"    [DEBUG] Found actual def at line {actual_def_line}: '{def_line[:60]}'")
        print(f"    [DEBUG] base_indent={base_indent}")

        # Scan from AFTER the actual def line
        for i in range(actual_def_line + 1, total_lines):
            line = lines[i]
            stripped = line.lstrip()

            # Skip blank lines
            if not stripped:
                continue

            indent = len(line) - len(line.lstrip(" \t"))

            # Found next top-level block?
            if indent <= base_indent:
                if (stripped.startswith("def ") or
                        stripped.startswith("async def ") or
                        stripped.startswith("class ") or
                        stripped.startswith("@")):
                    end_pos = sum(len(lines[j]) + 1 for j in range(i))
                    print(f"    [DEBUG] FOUND BOUNDARY at line {i}: {stripped[:60]}")
                    print(f"    [DEBUG] Method length: {end_pos - start_pos} chars")

                    return end_pos

        # No boundary found - EOF
        end_pos = len(content)
        print(f"    [DEBUG] NO BOUNDARY (EOF), method length: {end_pos - start_pos} chars")
        return end_pos

    def extract_all_methods(self):
        """Extract all target methods from multiple source files"""
        self.extracted_methods = []

        for file_config in self.target_files:
            source_path = Path(file_config['file'])
            # ✅ NEW: Check mode (default to 'methods' for backward compatibility)
            mode = file_config.get('mode', 'methods')

            if not source_path.exists():
                print(f"Error: Target file not found: {source_path}")
                continue

            try:
                file_stat = source_path.stat()
                if file_stat.st_size > self.max_file_size:
                    print(
                        f"Error: File too large ({file_stat.st_size / 1024 / 1024:.1f}MB): {source_path}")
                    continue
            except (OSError, PermissionError) as e:
                print(f"Cannot access {source_path}: {e}")
                continue

            try:
                with open(source_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {source_path}: {e}")
                continue

            print(f"\nProcessing: {source_path}")
            print(f"Mode: {mode}")

            # ✅ NEW: Full file mode
            if mode == 'file':
                file_data = {
                    'name': source_path.name,  # Use filename as "method name"
                    'content': content,
                    'line_number': 1,
                    'char_count': len(content),
                    'source_file': str(source_path),
                    'is_full_file': True  # Flag for export formatting
                }
                self.extracted_methods.append(file_data)
                print(
                    f"  ✓ Full file extracted ({len(content):,} chars, {content.count(chr(10)) + 1} lines)")

            # Existing methods mode
            elif mode == 'methods':
                target_methods = file_config.get('methods', [])
                print(f"Target methods: {len(target_methods)}")

                for method_name in target_methods:
                    print(f"  Extracting: {method_name}...", end=' ')
                    method_data = self.extract_method(content, method_name)
                    if method_data:
                        method_data['source_file'] = str(source_path)
                        method_data['is_full_file'] = False
                        self.extracted_methods.append(method_data)
                        print(
                            f"✓ (line {method_data['line_number']}, {method_data['char_count']} chars)")
                    else:
                        print("✗ NOT FOUND")

            else:
                print(f"  ⚠ Unknown mode '{mode}' - skipping")

        print(f"\n{'=' * 60}")
        print(f"Total items extracted: {len(self.extracted_methods)}")
        print(f"{'=' * 60}")

    def export_all(self):
        """Main export function - outputs single file"""
        if not self.extracted_methods:
            print("No methods to export. Run extract_all_methods() first.")
            return

        output_filename = f"METHOD_SANDBOX_{self.timestamp}.md"
        output_path = self.obsidian_base / output_filename

        try:
            self.obsidian_base.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write(f"# Method Extraction Sandbox\n\n")
                f.write(f"**Extracted**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Total Items**: {len(self.extracted_methods)}\n")
                f.write(f"**Source Files**: {len(self.target_files)}\n\n")
                f.write("---\n\n")

                current_file = None
                for i, method_data in enumerate(self.extracted_methods):
                    # Add file header when source changes
                    if method_data['source_file'] != current_file:
                        current_file = method_data['source_file']
                        f.write(f"## Source: `{current_file}`\n\n")

                    # ✅ NEW: Different header for full file vs method
                    if method_data.get('is_full_file', False):
                        f.write(f"### Full File: `{method_data['name']}`\n\n")
                        f.write(f"**Size**: {method_data['char_count']:,} characters\n\n")
                    else:
                        f.write(f"### Method: `{method_data['name']}()`\n\n")
                        f.write(f"**Line Number**: {method_data['line_number']}\n")
                        f.write(f"**Size**: {method_data['char_count']:,} characters\n\n")

                    # Auto-detect language from file extension
                    ext = Path(current_file).suffix.lower()
                    lang_map = {'.py': 'python', '.js': 'javascript', '.yml': 'yaml',
                                '.yaml': 'yaml', '.json': 'json'}
                    lang = lang_map.get(ext, 'text')

                    f.write(f"```{lang}\n")
                    f.write(method_data['content'])
                    f.write("\n```\n\n")
                    f.write("---\n\n")

            print(f"\n{'=' * 60}")
            print(f"✓ Export complete: {len(self.extracted_methods)} items")
            print(f"✓ Output file: {output_path}")
            print(f"{'=' * 60}")

        except Exception as e:
            print(f"Error writing output file: {e}")


# Usage
if __name__ == "__main__":
    config_path = "method_export_config.json"

    if Path(config_path).exists():
        extractor = MethodExtractor(config_path)
        extractor.extract_all_methods()
        extractor.export_all()
    else:
        sample_config = {
            "export_base": "C:/Users/wantless/Documents/Obsidian Vault/code_exports",
            "target_files": [
                {
                    "file": "./my_app/static/player.js",
                    "methods": ["calculateCanvasCoordinates", "renderPage"]
                },
                {
                    "file": "./my_app/pdf_processor/process.py",
                    "methods": ["get_citation_at_timestamp"]
                }
            ],
            "max_file_size_mb": 10
        }

        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=4)

        print(f"No config found. Created sample at: {config_path}")
        print("Please edit the config file and run again.")