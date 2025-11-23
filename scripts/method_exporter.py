# method_extractor.py - Extract specific methods from JavaScript file
import re
from pathlib import Path
from datetime import datetime


class JavaScriptMethodExtractor:
    def __init__(self, source_file: str, output_dir: str):
        self.source_file = Path(source_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_method(self, content: str, method_name: str) -> str:
        """Extract a complete method from JavaScript content"""

        # Pattern to match method definition (regular method or arrow function)
        patterns = [
            # Regular method: methodName(...) {
            rf'^\s*{re.escape(method_name)}\s*\([^)]*\)\s*\{{',
            # Arrow function property: methodName = (...) => {
            rf'^\s*{re.escape(method_name)}\s*=\s*(?:async\s*)?\([^)]*\)\s*=>\s*\{{',
            # Async method: async methodName(...) {
            rf'^\s*async\s+{re.escape(method_name)}\s*\([^)]*\)\s*\{{'
        ]

        for pattern in patterns:
            regex = re.compile(pattern, re.MULTILINE)
            match = regex.search(content)

            if match:
                start_pos = match.start()

                # Find the complete method by counting braces
                brace_count = 0
                in_method = False
                end_pos = start_pos

                for i in range(start_pos, len(content)):
                    char = content[i]

                    if char == '{':
                        brace_count += 1
                        in_method = True
                    elif char == '}':
                        brace_count -= 1

                    if in_method and brace_count == 0:
                        end_pos = i + 1
                        break

                method_content = content[start_pos:end_pos]
                return method_content

        return None

    def extract_methods(self, method_names: list) -> dict:
        """Extract multiple methods and return as dictionary"""

        if not self.source_file.exists():
            print(f"Error: Source file not found: {self.source_file}")
            return {}

        try:
            with open(self.source_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return {}

        extracted = {}
        for method_name in method_names:
            print(f"Extracting: {method_name}...", end=' ')
            method_code = self.extract_method(content, method_name)

            if method_code:
                extracted[method_name] = method_code
                print("✓")
            else:
                print("✗ NOT FOUND")

        return extracted

    def write_sandbox_file(self, methods_dict: dict, filename: str = None):
        """Write extracted methods to a single markdown file"""

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"method_sandbox_{timestamp}.md"

        output_path = self.output_dir / filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# JavaScript Method Sandbox - Coordinate Debug\n\n")
                f.write(f"**Source**: `{self.source_file}`\n")
                f.write(f"**Extracted**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Methods**: {len(methods_dict)}\n\n")
                f.write("---\n\n")

                for method_name, method_code in methods_dict.items():
                    f.write(f"## `{method_name}()`\n\n")
                    f.write("```javascript\n")
                    f.write(method_code)
                    f.write("\n```\n\n")
                    f.write("---\n\n")

            print(f"\n✓ Sandbox file created: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error writing sandbox file: {e}")
            return None

# ==============================
# =========== CONFIG ===========
# ==============================

# Usage
if __name__ == "__main__":
    # Configuration
    SOURCE_FILE = "./my_app/static/player.js"
    OUTPUT_DIR = "/mnt/c/Users/wantless/Documents/Obsidian Vault/code_exports"

    # Methods needed for coordinate debugging
    TARGET_METHODS = [
        "calculateCanvasCoordinates",
        "renderPage",
        "highlightAtTimestamp",
        "_aggregateSpanBounds"
    ]

    # Extract and export
    extractor = JavaScriptMethodExtractor(SOURCE_FILE, OUTPUT_DIR)
    methods = extractor.extract_methods(TARGET_METHODS)

    if methods:
        extractor.write_sandbox_file(methods, "debug_sandbox.md")
        print(f"\n{'=' * 60}")
        print(f"Extraction complete: {len(methods)}/{len(TARGET_METHODS)} methods")
        print(f"{'=' * 60}")
    else:
        print("No methods extracted. Check file path and method names.")