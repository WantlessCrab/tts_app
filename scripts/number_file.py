"""
Line Number Tool for Code Review
Adds line numbers to files for accurate reference tracking
"""

import os
from pathlib import Path

# ============================================
# CONFIGURATION (Edit this section only)
# ============================================

# Target file path (use raw string or forward slashes)
INPUT_FILE = Path(
    "/mnt/c/Users/admin/Documents/ResearchProjectVault/PycharmProProjectVault/Rebuild Roadmap/tts_app (MO2_MO3) Combination/player.js.md")

# Output options
CREATE_BACKUP = True  # Creates .bak file before numbering
OVERWRITE_ORIGINAL = False  # If True, replaces original file
OUTPUT_SUFFIX = "_numbered"  # Suffix for new file (if not overwriting)


# ============================================
# END CONFIGURATION
# ============================================


def add_line_numbers(input_path, output_path=None, create_backup=True):
    """
    Adds line numbers to the beginning of each line in a file.

    Args:
        input_path: Path to input file
        output_path: Path to output file (None = overwrite original)
        create_backup: If True and overwriting, creates .bak file first

    Returns:
        tuple: (success: bool, message: str, line_count: int)
    """
    input_file = Path(input_path)

    # Validation
    if not input_file.exists():
        return False, f"Error: File not found: {input_file}", 0

    if not input_file.is_file():
        return False, f"Error: Path is not a file: {input_file}", 0

    # Determine output path
    if output_path is None:
        output_file = input_file
        temp_file = input_file.with_suffix('.tmp')
    else:
        output_file = Path(output_path)
        temp_file = output_file

    # Create backup if requested and overwriting
    if create_backup and output_path is None:
        backup_file = input_file.with_suffix(input_file.suffix + '.bak')
        try:
            import shutil
            shutil.copy2(input_file, backup_file)
            print(f"✓ Backup created: {backup_file.name}")
        except Exception as e:
            return False, f"Backup failed: {e}", 0

    # Process file
    try:
        line_count = 0

        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
            line_count = len(lines)

        with open(temp_file, 'w', encoding='utf-8') as outfile:
            for index, line in enumerate(lines, 1):
                # Format: "1: content\n"
                numbered_line = f"{index}: {line}"
                outfile.write(numbered_line)

        # If overwriting, replace original with temp
        if output_path is None:
            import shutil
            shutil.move(str(temp_file), str(output_file))

        return True, f"Successfully numbered {line_count} lines", line_count

    except UnicodeDecodeError:
        return False, "Error: File encoding not UTF-8. Try different encoding.", 0
    except Exception as e:
        return False, f"Processing error: {e}", 0


def main():
    """Main execution function."""

    print("=" * 60)
    print("LINE NUMBER INSERTION TOOL")
    print("=" * 60)
    print()

    input_path = Path(INPUT_FILE)

    # Determine output path based on configuration
    if OVERWRITE_ORIGINAL:
        output_path = None
        action = "Overwriting original file"
    else:
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.with_name(f"{stem}{OUTPUT_SUFFIX}{suffix}")
        action = f"Creating numbered file: {output_path.name}"

    print(f"Input:  {input_path}")
    print(f"Action: {action}")
    print()

    # Execute
    success, message, line_count = add_line_numbers(
        input_path,
        output_path,
        CREATE_BACKUP
    )

    # Report results
    print("-" * 60)
    if success:
        print(f"✓ SUCCESS: {message}")
        print(f"✓ Output: {output_path if output_path else input_path}")
        print(f"✓ Total lines numbered: {line_count}")
    else:
        print(f"✗ FAILED: {message}")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())