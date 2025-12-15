import os
import json

# --- CONFIGURATION ---
# The folder you want to scan (Your WSL Path)
TARGET_DIR = "/home/wantless/TTS/feature_llm_bridge"

def generate_terminal_output():
    print(f"--- SCANNING: {TARGET_DIR} ---\n")
    print("Copy the block below and paste it INSIDE your 'target_files': [\n")

    # 1. Walk the directory
    for root, dirs, files in os.walk(TARGET_DIR):
        for file in files:
            # 2. Filter out junk
            if file.startswith(".") or "__pycache__" in root or file.endswith(".pyc"):
                continue

            # 3. Construct the absolute path
            # We use .replace() to ensure JSON-friendly forward slashes if preferred,
            # though json.dumps handles backslashes correctly too.
            full_path = os.path.join(root, file).replace("\\", "/")

            # 4. Create the dictionary object
            entry = {
                "file": full_path,
                "mode": "file",
                "comment": f"Auto-detected: {file}"
            }

            # 5. Print the JSON object followed by a comma
            # The indent=2 makes it look exactly like your example
            print(json.dumps(entry, indent=2) + ",")

    print("\n] <--- End of paste block")


if __name__ == "__main__":
    generate_terminal_output()