import ast
import os
import sys

# =============================================================================
# 🎯 CONFIGURATION: POINT AND SHOOT
# =============================================================================
# Paste your project-standard configuration objects here.
TARGET_FILES = [
    {
        "file": "../my_app/pdf_processor/extraction_engine.py",
        "mode": "path",
        "comment": "browser extension app"
    },
    {
        "file": "../my_app/pdf_processor/process.py",
        "mode": "file",
        "comment": "Orchestrator"
    }
]


# =============================================================================

def get_definition_header(node, source_lines):
    """
    Extracts ONLY the definition lines (signature) of a node.
    """
    start_line = node.lineno - 1
    header_lines = []

    for i in range(start_line, len(source_lines)):
        line = source_lines[i]
        header_lines.append(line.rstrip())
        clean_line = line.split('#')[0].strip()
        if clean_line.endswith(":"):
            break

    return "\n".join(header_lines)


def analyze_file(target_config):
    # Support both simple strings and dict objects
    if isinstance(target_config, dict):
        file_path = target_config.get("file")
        comment = target_config.get("comment", "")
    else:
        file_path = target_config
        comment = ""

    # Resolve path relative to this script if not absolute
    if not os.path.isabs(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_path)

    print(f"\n{'=' * 80}")
    print(f"📄 TARGET: {os.path.basename(file_path)}")
    print(f"📂 PATH:   {file_path}")
    if comment:
        print(f"💬 NOTE:   {comment}")
    print(f"{'=' * 80}\n")

    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found: {file_path}")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
            source_lines = source.splitlines()
            tree = ast.parse(source)
    except Exception as e:
        print(f"❌ ERROR: Could not parse file. {e}")
        return

    for node in tree.body:
        # 1. Top-Level Constants
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            print(source_lines[node.lineno - 1].strip())

        # 2. Top-Level Functions
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            print(f"\n{get_definition_header(node, source_lines)}")

        # 3. Classes
        elif isinstance(node, ast.ClassDef):
            print(f"\n{get_definition_header(node, source_lines)}")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_sig = get_definition_header(item, source_lines)
                    # Indent methods for visual hierarchy
                    indented_sig = "\n".join("    " + line for line in method_sig.splitlines())
                    print(indented_sig)


if __name__ == "__main__":
    if not TARGET_FILES:
        print("⚠️ No targets configured. Add files to TARGET_FILES list at the top.")
    else:
        for target in TARGET_FILES:
            analyze_file(target)