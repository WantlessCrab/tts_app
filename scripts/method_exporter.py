# method_exporter.py
import os
import re
import json
import random
import string
from pathlib import Path
from datetime import datetime
from typing import Set, Dict, List, Optional


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
            # Python function: def method_name(...):
            rf'^\s*def\s+{re.escape(method_name)}\s*\([^)]*\)\s*:',
            # Python async: async def method_name(...):
            rf'^\s*async\s+def\s+{re.escape(method_name)}\s*\([^)]*\)\s*:',
            # JavaScript regular method
            rf'^\s*{re.escape(method_name)}\s*\([^)]*\)\s*\{{',
            # JavaScript arrow function
            rf'^\s*{re.escape(method_name)}\s*=\s*(?:async\s*)?\([^)]*\)\s*=>\s*\{{',
            # JavaScript async method
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
        """Find end of JavaScript method by counting braces"""
        brace_count = 0
        in_method = False
        in_string = False
        in_regex = False
        escape_next = False
        string_char = None

        i = start_pos
        while i < len(content):
            char = content[i]

            # Handle escape sequences
            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == '\\':
                escape_next = True
                i += 1
                continue

            # Handle strings (avoid counting braces inside strings)
            if not in_regex:
                if char in ('"', "'", '`') and not in_string:
                    in_string = True
                    string_char = char
                    i += 1
                    continue
                elif in_string and char == string_char:
                    in_string = False
                    string_char = None
                    i += 1
                    continue

            # If we're in a string, skip brace counting
            if in_string:
                i += 1
                continue

            # Count braces
            if char == '{':
                brace_count += 1
                in_method = True
            elif char == '}':
                brace_count -= 1
                if in_method and brace_count == 0:
                    return i + 1

            i += 1

        # EOF reached
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

        # ✅ CHANGE: Loop through each file config
        for file_config in self.target_files:
            source_path = Path(file_config['file'])
            target_methods = file_config['methods']

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

            print(f"\nExtracting from: {source_path}")
            print(f"Target methods: {len(target_methods)}")

            # Extract methods from this file
            for method_name in target_methods:
                print(f"  Extracting: {method_name}...", end=' ')
                method_data = self.extract_method(content, method_name)

                if method_data:
                    # ✅ CHANGE: Add source file to metadata
                    method_data['source_file'] = str(source_path)
                    self.extracted_methods.append(method_data)
                    print(
                        f"✓ (line {method_data['line_number']}, {method_data['char_count']} chars)")
                else:
                    print("✗ NOT FOUND")

        print(f"\n{'=' * 60}")
        print(f"Total methods extracted: {len(self.extracted_methods)}")
        print(f"{'=' * 60}")

    def export_all(self):
        """Main export function - outputs single file"""
        if not self.extracted_methods:
            print("No methods to export. Run extract_all_methods() first.")
            return

        # Create single output file (not directory)
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
                f.write(f"**Total Methods**: {len(self.extracted_methods)}\n")
                f.write(f"**Source Files**: {len(self.target_files)}\n\n")
                f.write("---\n\n")

                # Write each method (grouped by file)
                current_file = None
                for i, method_data in enumerate(self.extracted_methods):
                    # ✅ CHANGE: Add file header when source changes
                    if method_data['source_file'] != current_file:
                        current_file = method_data['source_file']
                        f.write(f"## Source: `{current_file}`\n\n")

                    f.write(f"### Method: `{method_data['name']}()`\n\n")
                    f.write(f"**Line Number**: {method_data['line_number']}\n")
                    f.write(f"**Size**: {method_data['char_count']:,} characters\n\n")

                    # ✅ CHANGE: Auto-detect language from file extension
                    ext = Path(current_file).suffix.lower()
                    lang = 'python' if ext == '.py' else 'javascript' if ext == '.js' else 'text'

                    f.write(f"```{lang}\n")
                    f.write(method_data['content'])
                    f.write("\n```\n\n")
                    f.write("---\n\n")

            print(f"\n{'=' * 60}")
            print(f"✓ Export complete: {len(self.extracted_methods)} methods")
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
            "export_base": "C:/Users/wantl/Documents/Obsidian Vault/code_exports",
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