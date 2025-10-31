# code_exporter_full.py (FINAL, v1.8)
import os
import re
import json
import fnmatch
import traceback
import random
import string
from pathlib import Path
from datetime import datetime
from typing import Set, Dict, List, Optional


class CodeProjectExporter:
    def __init__(self, config_path="export_config.json"):
        self.config_path = config_path

        self.obsidian_base: Optional[Path] = None
        self.projects: List[Dict] = []
        self.global_include_patterns: List[str] = []
        self.global_skip_dirs: Set[str] = set()
        self.global_exclude_patterns: List[str] = []
        self.max_file_size: int = 0

        self.load_config()

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + ''.join(
            random.choices(string.ascii_lowercase + string.digits, k=8))
        self.exported_files: List[Dict] = []

        self._global_compiled_patterns = [
            re.compile(fnmatch.translate(pattern), re.IGNORECASE)
            for pattern in self.global_include_patterns
        ]
        self._global_compiled_exclude_patterns = [
            re.compile(fnmatch.translate(pattern), re.IGNORECASE)
            for pattern in self.global_exclude_patterns
        ]

    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            self.obsidian_base = Path(config.get("export_base"))
            self.projects = config.get("projects", [])
            self.global_include_patterns = config.get("global_include_patterns", [])
            self.global_skip_dirs = set(config.get("global_skip_dirs", []))
            self.global_exclude_patterns = config.get("global_exclude_patterns", [])
            self.max_file_size = config.get("max_file_size_mb", 5) * 1024 * 1024

            if not self.obsidian_base or not self.projects:
                raise KeyError("Config missing 'export_base' or 'projects'")

        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Stopping.")
            exit()
        except json.JSONDecodeError:
            print(f"Error: Config file {self.config_path} is not valid JSON. Stopping.")
            exit()
        except KeyError as e:
            print(f"Error: Config file missing required key: {e}. Stopping.")
            exit()

    # --- v1.8 Change ---
    def create_export_summary(self, results):
        """Create a summary file for all exports, including trees"""
        summary_path = self.obsidian_base / f"EXPORT_SUMMARY_{self.timestamp}.md"

        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"# Code Export Summary\n\n")
                f.write(f"**Timestamp**: {self.timestamp}\n")
                f.write(f"**Total Projects**: {len(results)}\n\n")

                f.write("## Export Results\n\n")
                for result in results:
                    status = "✓" if result['success'] else "✗"
                    f.write(f"- {status} **{result['name']}**\n")
                    f.write(f"  - Source: `{result['path']}`\n")
                    if result['output']:
                        link_path = f"{result['output'].name}/00_PROJECT_INDEX"
                        f.write(f"  - Output: [[{link_path}|{result['name']} Project Index]]\n")
                    else:
                        f.write(f"  - Output: Failed to export (No files found or error)\n")
                    f.write("\n")

                # --- v1.8 New Section ---
                f.write("---\n")
                f.write("## Aggregated Project File Trees\n\n")
                for result in results:
                    if result['success']:
                        f.write(f"### {result['name']}\n\n")
                        f.write("```\n")
                        f.write(result['tree_content'])
                        f.write("\n```\n\n")

            print(f"\nSummary saved to: {summary_path}")
        except Exception as e:
            print(f"Could not create summary: {e}")

    def collect_matching_files(self, root_path: Path,
                               patterns: List[re.Pattern],
                               skip_dirs_set: Set[str],
                               exclude_patterns: List[re.Pattern]):
        """Recursively collect all files matching provided patterns"""
        print(f"Scanning {root_path}...")
        self.exported_files = []
        file_count = 0

        for item in root_path.rglob('*'):
            file_count += 1
            if file_count % 500 == 0:
                print(f"  ...scanned {file_count} items".ljust(50), end='\r')

            if item.is_symlink():
                continue

            if any(part == skip_dir for part in item.parts for skip_dir in skip_dirs_set):
                continue

            if item.is_file():
                try:
                    if item.stat().st_size > self.max_file_size:
                        print(f"  Skipping large file ({item.stat().st_size / 1024 / 1024:.1f}MB): {item.name}")
                        continue

                    if self._matches_patterns(item.name, patterns):
                        if self._matches_patterns(item.name, exclude_patterns):
                            continue  # File is explicitly excluded

                        self.exported_files.append({
                            'full_path': item,
                            'relative_path': item.relative_to(root_path)
                        })
                except (OSError, PermissionError) as e:
                    print(f"  Cannot access {item.name}: {e}")
                    continue

        print(f"\nScan complete. Found {len(self.exported_files)} matching files.".ljust(50))

    def _matches_patterns(self, filename: str, patterns: List[re.Pattern]) -> bool:
        """Check if filename matches any of the provided patterns"""
        return any(pattern.match(filename) for pattern in patterns)

    def _render_tree(self, tree_dict: Dict, root_name: str, prefix: str = "") -> str:
        """Render tree dictionary as string"""
        lines = []
        if not prefix:
            lines.append(f"{root_name}/")

        items = sorted(tree_dict.items(), key=lambda x: (x[1] is None, x[0].lower()))

        for i, (name, subtree) in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "

            if subtree is None:
                lines.append(f"{prefix}{connector}{name}")
            else:
                lines.append(f"{prefix}{connector}{name}/")
                extension = "    " if is_last else "│   "
                subtree_lines = self._render_tree(subtree, name, prefix + extension)
                lines.extend(subtree_lines.splitlines())

        return "\n".join(filter(None, lines))

    def export_file_with_context(self, file_path, relative_path, output_dir):
        """Export a single file maintaining directory context"""
        source_file = Path(file_path)

        try:
            file_stat = source_file.stat()
            if file_stat.st_size > self.max_file_size:
                print(f"  Skipping large file during export: {relative_path}")
                return
            file_size_str = f"{file_stat.st_size:,} bytes"
            file_mod_time = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

        except (OSError, PermissionError) as e:
            print(f"  Cannot access {file_path} during export: {e}")
            return

        safe_name = str(relative_path).replace(os.sep, '_')
        safe_name = re.sub(r'[<>:"|?*]', '_', safe_name)

        if len(safe_name) > 200:
            safe_name = safe_name[:197] + "..."

        md_filename = f"{safe_name}.md"
        output_path = output_dir / md_filename

        try:
            with open(source_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return

        ext = source_file.suffix.lower()
        lang_map = {
            '.py': 'python', '.yml': 'yaml', '.yaml': 'yaml', '.json': 'json',
            '.html': 'html', '.js': 'javascript', '.css': 'css', '.sh': 'bash',
            '.md': 'markdown',
        }

        if source_file.name.lower().startswith('dockerfile'):
            language = 'dockerfile'
        else:
            language = lang_map.get(ext, 'text')

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# {relative_path}\n\n")
                f.write(f"**Full Path**: `{source_file}`\n")
                f.write(f"**Size**: {file_size_str}\n")
                f.write(f"**Modified**: {file_mod_time}\n\n")
                f.write("---\n\n")
                f.write(f"```{language}\n")
                f.write(content)
                f.write("\n```\n")
        except Exception as e:
            print(f"Error writing {output_path}: {e}")

    def export_all_projects(self):
        """Export all projects defined in config"""
        if not self.projects:
            print("No projects defined in config.")
            return

        results = []
        for project in self.projects:
            print(f"\n{'=' * 60}")
            print(f"Exporting: {project.get('name', 'Unknown')}")
            print(f"{'=' * 60}")

            # --- Select Patterns ---
            project_patterns = project.get("include_patterns")
            if project_patterns:
                print(f"  Using per-project include patterns...")
                compiled_patterns = [
                    re.compile(fnmatch.translate(p), re.IGNORECASE)
                    for p in project_patterns
                ]
            else:
                print(f"  Using global include patterns...")
                compiled_patterns = self._global_compiled_patterns

            # --- Select Skip Dirs ---
            project_skip_dirs = project.get("skip_dirs")
            if project_skip_dirs is not None:
                print(f"  Using per-project skip dirs...")
                skip_dirs_set = set(project_skip_dirs)
            else:
                print(f"  Using global skip dirs...")
                skip_dirs_set = self.global_skip_dirs

            # --- Select Exclude Patterns ---
            project_exclude_patterns = project.get("exclude_patterns")
            if project_exclude_patterns:
                print(f"  Using per-project exclude patterns...")
                compiled_exclude_patterns = [
                    re.compile(fnmatch.translate(p), re.IGNORECASE)
                    for p in project_exclude_patterns
                ]
            else:
                print(f"  Using global exclude patterns...")
                compiled_exclude_patterns = self._global_compiled_exclude_patterns

            result = self.export_directory(
                project.get('path'),
                project.get('name'),
                compiled_patterns,
                skip_dirs_set,
                compiled_exclude_patterns
            )

            # --- v1.8 Change ---
            results.append({
                'name': project.get('name'),
                'path': project.get('path'),
                'output': result.get("output_dir") if result else None,
                'tree_content': result.get("tree_content") if result else "No files found or error.",
                'success': result is not None
            })

        self.create_export_summary(results)
        return results

    def export_directory(self, source_dir: str, project_name: str,
                         patterns: List[re.Pattern],
                         skip_dirs_set: Set[str],
                         exclude_patterns: List[re.Pattern]):
        """Export entire directory structure to Obsidian"""
        if not source_dir:
            print("Error: Project path is empty in config.")
            return None

        source_path = Path(source_dir)

        if not source_path.exists():
            print(f"Error: Directory not found: {source_dir}")
            return None

        if project_name is None:
            project_name = source_path.name

        output_dir = self.obsidian_base / f"{project_name}_{self.timestamp}"

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return None

        self.collect_matching_files(source_path, patterns, skip_dirs_set, exclude_patterns)

        if not self.exported_files:
            print("  No matching files found. Aborting export for this project.")
            try:
                output_dir.rmdir()
            except OSError as e:
                print(f"Warning: Could not remove empty dir {output_dir}: {e}")
            return None

        self.exported_files.sort(key=lambda x: str(x['relative_path']).lower())

        tree_content = self.generate_filtered_tree(source_path)

        self.write_index_file(output_dir, project_name, source_path, tree_content)
        self.create_directory_structure(source_path, output_dir)

        exported_count = 0
        for file_info in self.exported_files:
            try:
                self.export_file_with_context(
                    file_info['full_path'],
                    file_info['relative_path'],
                    output_dir
                )
                exported_count += 1
            except Exception as e:
                print(f"Error exporting {file_info['relative_path']}: {e}")

        print(f"✓ Exported {exported_count}/{len(self.exported_files)} files to: {output_dir}")

        # --- v1.8 Change ---
        return {
            "output_dir": output_dir,
            "tree_content": tree_content
        }

    def generate_filtered_tree(self, root_path) -> str:
        """Generate tree showing only matched files and their parent directories"""
        tree_dict = self._build_tree_dict()
        return self._render_tree(tree_dict, root_path.name)

    def _build_tree_dict(self) -> Dict:
        """Build nested dictionary structure from the file list"""
        tree = {}

        for file_info in self.exported_files:
            parts = file_info['relative_path'].parts
            current = tree

            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = None
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        return tree

    def write_index_file(self, output_dir, project_name, source_path, tree_content):
        """Write the main index file"""
        index_path = output_dir / "00_PROJECT_INDEX.md"
        try:
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(f"# {project_name} Export\n\n")
                f.write(f"**Source**: `{source_path}`\n")
                f.write(f"**Exported**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Total Files**: {len(self.exported_files)}\n")
                f.write(f"**Max File Size**: {self.max_file_size / 1024 / 1024:.1f}MB\n\n")
                f.write("## Project Structure (Filtered)\n\n")
                f.write("```\n")
                f.write(tree_content)
                f.write("\n```\n\n")
                f.write("## Exported Files\n\n")

                current_dir = None
                for file_info in self.exported_files:
                    rel_path = file_info['relative_path']
                    file_dir = rel_path.parent

                    if file_dir != current_dir:
                        current_dir = file_dir
                        f.write(f"\n### {file_dir if str(file_dir) != '.' else 'Root'}\n\n")

                    safe_link = str(rel_path).replace(os.sep, '_')
                    f.write(f"- [[{safe_link}.md|{rel_path.name}]]\n")
        except Exception as e:
            print(f"Error writing index file: {e}")
            traceback.print_exc()

    def create_directory_structure(self, source_path, output_dir):
        """Create directory README stubs to preserve structure in Obsidian"""
        dirs_created = set()

        for file_info in self.exported_files:
            rel_path = file_info['relative_path']
            dir_path = rel_path.parent

            if str(dir_path) != '.' and dir_path not in dirs_created:
                readme_name = f"_DIR_{str(dir_path).replace(os.sep, '_')}_README.md"
                readme_path = output_dir / readme_name

                try:
                    with open(readme_path, 'w', encoding='utf-8') as f:
                        f.write(f"# Directory: {dir_path}\n\n")
                        f.write(f"This directory contains files from `{source_path / dir_path}`\n\n")
                        f.write("## Files in this directory\n\n")

                        for fi in self.exported_files:
                            if fi['relative_path'].parent == dir_path:
                                safe_link = str(fi['relative_path']).replace(os.sep, '_')
                                f.write(f"- [[{safe_link}.md|{fi['relative_path'].name}]]\n")

                        dirs_created.add(dir_path)
                except Exception as e:
                    print(f"Warning: Could not create README for {dir_path}: {e}")


# Usage
if __name__ == "__main__":
    config_path = "export_config.json"

    if Path(config_path).exists():
        exporter = CodeProjectExporter(config_path)
        exporter.export_all_projects()
    else:
        sample_config = {
            "export_base": "C:/path/to/your/obsidian/exports",
            "projects": [
                {
                    "path": "C:/path/to/your/project-A",
                    "name": "Project_A_Snapshot"
                }
            ],
            "global_include_patterns": [
                "*.py", "*.yml", "*.yaml", "*.json", "Dockerfile*", "*.md"
            ],
            "global_skip_dirs": ["__pycache__", ".git", ".idea", "venv", ".venv"],
            "global_exclude_patterns": ["README.md"],
            "max_file_size_mb": 5
        }

        with open("export_config.json", 'w') as f:
            json.dump(sample_config, f, indent=4)

        print(f"No config found. Created sample at: {config_path}")
        print("Please edit it to point to your project paths.")