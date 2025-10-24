import csv
import os
import re
import shutil
from glob import glob


# Folder containing the CSV files to update
FOLDER = "/Users/piyushwanchoo/Documents/Post_Doc/1000_RUN_SHOTS/meta_data"


# Map JHAMAB code to material
CODE_TO_MATERIAL = {
    "19": "Cu",
    "20": "Zn",
    "21": "Brass",
}


# Regex to extract the JHAMAB code portion (19, 20, 21) from the filename
JHAMAB_REGEX = re.compile(r"JHAMAB000(19|20|21)-", re.IGNORECASE)


def infer_material_from_filename(filename: str):
    basename = os.path.basename(filename)
    m = JHAMAB_REGEX.search(basename)
    if not m:
        return None
    return CODE_TO_MATERIAL.get(m.group(1))


def update_sample_material(csv_path: str, material: str):
    # Read the CSV
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        rows = list(reader)

    # Ensure the header contains "Sample material"
    if "Sample material" not in fieldnames:
        fieldnames.append("Sample material")

    # Overwrite the column for every row
    for row in rows:
        row["Sample material"] = material

    # Write to a temp file, back up original, then replace
    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    backup_path = csv_path + ".bak"
    shutil.copy2(csv_path, backup_path)
    os.replace(tmp_path, csv_path)


def main():
    # Match the intended files
    pattern = os.path.join(FOLDER, "LMI_20251023_JHAMAB000*-*.csv")
    files = sorted(glob(pattern))
    if not files:
        print(f"No files matched: {pattern}")
        return

    for path in files:
        material = infer_material_from_filename(path)
        if not material:
            print(f"Skipping (no mapping): {path}")
            continue
        try:
            update_sample_material(path, material)
            print(f"Updated: {path} -> Sample material = {material}")
        except Exception as e:
            print(f"ERROR updating {path}: {e}")


if __name__ == "__main__":
    main()


