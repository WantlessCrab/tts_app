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
            self.target_files = config.get("target_files", [])
            self.max_file_size = config.get("max_file_size_mb", 10) * 1024 * 1024

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

    # --- EXTRACTION LOGIC ---

    def extract_method(self, content: str, method_name: str) -> Optional[Dict]:
        """Legacy: Extract a complete method from JavaScript OR Python content"""
        patterns = [
            # Python function
            rf'^\s*def\s+{re.escape(method_name)}\s*\([^)]*\)\s*:',
            # Python async
            rf'^\s*async\s+def\s+{re.escape(method_name)}\s*\([^)]*\)\s*:',
            # JS: Class method (e.g. render() {})
            rf'^\s*{re.escape(method_name)}\s*\([^)]*\)\s*\{{',
            # JS: Arrow function (e.g. const render = () => {})
            rf'^\s*(?:const|let|var)?\s*{re.escape(method_name)}\s*=\s*(?:async\s*)?\(?[^)]*\)?\s*=>\s*\{{',
            # JS: Async Class method
            rf'^\s*async\s+{re.escape(method_name)}\s*\([^)]*\)\s*\{{',
            # JS: Standard Function (CRITICAL ADDITION for content.js)
            rf'^\s*function\s+{re.escape(method_name)}\s*\([^)]*\)\s*\{{',
            # JS: Async Function (CRITICAL ADDITION for content.js)
            rf'^\s*async\s+function\s+{re.escape(method_name)}\s*\([^)]*\)\s*\{{'
        ]

        for pattern in patterns:
            regex = re.compile(pattern, re.MULTILINE)
            match = regex.search(content)

            if match:
                start_pos = match.start()

                # Determine language based on pattern signature
                is_python = 'def ' in pattern

                if is_python:
                    end_pos = self._find_python_method_block(content, start_pos)
                else:
                    end_pos = self._find_js_method_end(content, start_pos)

                method_content = content[start_pos:end_pos]
                line_number = content[:start_pos].count('\n') + 1

                return {
                    'name': method_name,
                    'type': 'method',  # Ensures compatibility with new export_all
                    'content': method_content,
                    'line_number': line_number,
                    'char_count': len(method_content)
                }
        return None

    def _extract_header(self, content: str) -> Dict:
        """Extracts imports and constants (top of file until first function/class)"""
        # Regex to find the first definition of a class or function
        patterns = [
            r'^\s*def\s+', r'^\s*async\s+def\s+', r'^\s*class\s+',  # Python
            r'^\s*function\s+', r'^\s*class\s+', r'^\s*const\s+\w+\s*=\s*\(.*=>',  # JS
        ]

        first_def_pos = len(content)

        for pattern in patterns:
            match = re.search(pattern, content, re.MULTILINE)
            if match and match.start() < first_def_pos:
                first_def_pos = match.start()

        header_content = content[:first_def_pos].strip()

        return {
            'name': 'FILE_HEADER (Imports/Consts)',
            'type': 'header',
            'content': header_content,
            'line_number': 1,
            'char_count': len(header_content)
        }

    def _extract_full_file(self, content: str, filename: str) -> Dict:
        """Extracts the entire file content"""
        return {
            'name': f'FULL_FILE: {filename}',
            'type': 'full_file',
            'content': content,
            'line_number': 1,
            'char_count': len(content)
        }

    def _extract_lines(self, content: str, lines_range: List[int]) -> Dict:
        """Extracts specific lines (1-based index). Format: [start, end]"""
        all_lines = content.split('\n')
        start, end = lines_range[0], lines_range[1]

        # Clamp to valid range
        start = max(1, start)
        end = min(len(all_lines), end)

        selected_lines = all_lines[start - 1:end]
        snippet = '\n'.join(selected_lines)

        return {
            'name': f'LINES {start}-{end}',
            'type': 'snippet',
            'content': snippet,
            'line_number': start,
            'char_count': len(snippet)
        }

    # --- HELPER LOGIC (Preserved from original) ---

    def _find_js_method_end(self, content: str, start_pos: int) -> int:
        brace_count = 0
        in_method = False
        in_string = False
        escape_next = False
        string_char = None
        i = start_pos
        while i < len(content):
            char = content[i]
            if escape_next:
                escape_next = False
                i += 1
                continue
            if char == '\\':
                escape_next = True
                i += 1
                continue
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
            if in_string:
                i += 1
                continue
            if char == '{':
                brace_count += 1
                in_method = True
            elif char == '}':
                brace_count -= 1
                if in_method and brace_count == 0:
                    return i + 1
            i += 1
        return len(content)

    def _find_python_method_block(self, content: str, start_pos: int) -> int:
        lines = content.split("\n")
        start_line_idx = content[:start_pos].count("\n")
        total_lines = len(lines)
        actual_def_line = None

        for i in range(start_line_idx, min(start_line_idx + 10, total_lines)):
            line = lines[i]
            stripped = line.lstrip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                actual_def_line = i
                break

        if actual_def_line is None: return len(content)

        def_line = lines[actual_def_line]
        base_indent = len(def_line) - len(def_line.lstrip(" \t"))

        for i in range(actual_def_line + 1, total_lines):
            line = lines[i]
            stripped = line.lstrip()
            if not stripped: continue
            indent = len(line) - len(line.lstrip(" \t"))
            if indent <= base_indent:
                if (stripped.startswith("def ") or stripped.startswith("async def ") or
                        stripped.startswith("class ") or stripped.startswith("@")):
                    end_pos = sum(len(lines[j]) + 1 for j in range(i))
                    return end_pos
        return len(content)

    # --- MAIN EXECUTION ---

    def extract_all_methods(self):
        """Main loop processing target files based on 'mode'"""
        self.extracted_methods = []

        for file_config in self.target_files:
            source_path = Path(file_config['file'])
            # Default to 'methods' mode if not specified
            mode = file_config.get('mode', 'methods')

            if not source_path.exists():
                print(f"Error: Target file not found: {source_path}")
                continue

            try:
                with open(source_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {source_path}: {e}")
                continue

            print(f"\nProcessing: {source_path.name} [Mode: {mode}]")

            # MODE SWITCHING LOGIC
            if mode == 'full':
                data = self._extract_full_file(content, source_path.name)
                data['source_file'] = str(source_path)
                self.extracted_methods.append(data)
                print(f"  ✓ Full file extracted ({data['char_count']} chars)")

            elif mode == 'header':
                data = self._extract_header(content)
                data['source_file'] = str(source_path)
                self.extracted_methods.append(data)
                print(f"  ✓ Header extracted ({data['char_count']} chars)")

            elif mode == 'lines':
                # Expecting "lines": [1, 50]
                ranges = file_config.get('lines', [])
                if ranges and len(ranges) == 2:
                    data = self._extract_lines(content, ranges)
                    data['source_file'] = str(source_path)
                    self.extracted_methods.append(data)
                    print(f"  ✓ Lines {ranges[0]}-{ranges[1]} extracted")
                else:
                    print("  ✗ Error: 'lines' mode requires a [start, end] array.")

            else:  # Default 'methods' mode
                target_methods = file_config.get('methods', [])
                for method_name in target_methods:
                    print(f"  Extracting: {method_name}...", end=' ')
                    method_data = self.extract_method(content, method_name)
                    if method_data:
                        method_data['source_file'] = str(source_path)
                        self.extracted_methods.append(method_data)
                        print(f"✓ (line {method_data['line_number']})")
                    else:
                        print("✗ NOT FOUND")

        print(f"\n{'=' * 60}")
        print(f"Total items extracted: {len(self.extracted_methods)}")
        print(f"{'=' * 60}")

    def export_all(self):
        if not self.extracted_methods:
            print("No content to export.")
            return

        output_filename = f"SIMULATION_{self.timestamp}.md"
        output_path = self.obsidian_base / output_filename

        try:
            self.obsidian_base.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# Simulation Output\n\n")
                f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("---\n\n")

                current_file = None
                for item in self.extracted_methods:
                    if item['source_file'] != current_file:
                        current_file = item['source_file']
                        f.write(f"## Source: `{current_file}`\n\n")

                    # Header Formatting
                    if item['type'] == 'method':
                        f.write(f"### Method: `{item['name']}()`\n")
                    elif item['type'] == 'header':
                        f.write(f"### Header (Imports & Constants)\n")
                    elif item['type'] == 'full_file':
                        f.write(f"### Full File Content\n")
                    elif item['type'] == 'snippet':
                        f.write(f"### {item['name']}\n")

                    f.write(
                        f"**Line**: {item['line_number']} | **Size**: {item['char_count']:,} chars\n\n")

                    ext = Path(current_file).suffix.lower()
                    lang = 'python' if ext == '.py' else 'javascript' if ext == '.js' else 'text'
                    if ext == '.css': lang = 'css'
                    if ext == '.json': lang = 'json'

                    f.write(f"```{lang}\n")
                    f.write(item['content'])
                    f.write("\n```\n\n")
                    f.write("---\n\n")

            print(f"✓ Export saved to: {output_path}")

        except Exception as e:
            print(f"Error writing file: {e}")


if __name__ == "__main__":
    if Path("method_export_config.json").exists():
        extractor = MethodExtractor()
        extractor.extract_all_methods()
        extractor.export_all()
    else:
        print("Config not found.")