# Directory data contains various subdirectories, and each of them has
# a subdirectory named after the seed value (for example, data/xxx/1967)
# The directory "human" shares the same structure.
# For each subdirectory, replace the file named "selected_reviews.csv" in "data"
# # with the file named "selected_reviews.csv" in "human".
import os
import shutil
import argparse

def replace_selected_reviews(data_dir: str, human_dir: str, seed: int) -> int:
    """
    For each subdirectory in 'data_dir', check if same same exists in 'human_dir'.
    Then, opens selected_reviews.csv in the human dir and, for each review,
    replaces the 'hscore' value with the corresponding 'hscore' value from the
    'selected_reviews.csv' in 'data_dir'.

    Args:
        data_dir (str): The path to the data directory containing subdirectories.
        human_dir (str): The path to the human directory containing subdirectories.
        seed (int): The seed value to specify the subdirectory.

    Returns:
        int: The number of replaced rows across all subdirectories.
    """
    # Iterate only directory entries in data_dir (skip files). Use sorted() for
    # deterministic ordering across runs.
    replacedItems: int = 0

    for subdir in sorted(os.listdir(data_dir)):
        entry_path: str = os.path.join(data_dir, subdir)
        if not os.path.isdir(entry_path):
            # Skip files that may exist in the data directory (e.g., CSVs, pkl)
            continue
        reviewsFileName: str = os.path.join(entry_path, str(seed), 'selected_reviews.csv')
        if not os.path.exists(reviewsFileName):
            print(f"File {reviewsFileName} does not exist, skipping replacement.")
            continue
        humanSubdir: str = os.path.join(human_dir, subdir, str(seed))
        humanScoresFileName: str = os.path.join(humanSubdir, 'selected_reviews.csv')
        if not os.path.exists(humanScoresFileName):
            print(f"Human file {humanScoresFileName} does not exist, skipping replacement.")
            continue

        replacedItems += replaceHscoreInFiles(reviewsFileName, humanScoresFileName)
    return replacedItems

def replaceHscoreInFiles(target_csv: str, source_csv: str, key_field: str | None = None) -> int:
    """
    Replace the 'hscore' values in `target_csv` with values from `source_csv`.

    Matching is done by a key field. If `key_field` is provided it must be one of
    the column names in the CSVs. If not provided the function will attempt to
    match by the 'review' column first, then by 'readable'.

    Both CSVs are expected to have a header with at least the columns
    'readable', 'hscore', 'review'. Returns the number of replaced rows.
    """
    import csv

    # Helper to read CSV into list of dicts
    def _read_csv(path: str) -> list[dict]:
        with open(path, 'r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            return list(reader)

    # Read source and target
    src_rows = _read_csv(source_csv)
    tgt_rows = _read_csv(target_csv)

    if not src_rows or not tgt_rows:
        raise ValueError('Source or target CSV is empty')

    # Determine key field
    fields = list(tgt_rows[0].keys())
    if key_field:
        if key_field not in fields:
            raise ValueError(f"Key field '{key_field}' not found in target CSV")
        key = key_field
    else:
        key = 'review' if 'review' in fields else ('readable' if 'readable' in fields else None)
        if key is None:
            raise ValueError('No suitable key field found in target CSV (need review or readable)')

    # Build lookup from source
    src_lookup: dict[str, str] = {}
    for r in src_rows:
        k = r.get(key, '').strip()
        if not k:
            continue
        src_lookup[k] = r.get('hscore', '').strip()

    # Replace in target
    replaced = 0
    for r in tgt_rows:
        k = r.get(key, '').strip()
        if not k:
            continue
        new_score = src_lookup.get(k)
        if new_score is not None and new_score != r.get('hscore', '').strip():
            r['hscore'] = new_score
            replaced += 1

    # Write back target CSV (preserve field order)
    with open(target_csv, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tgt_rows)

    return replaced

if __name__ == "__main__":

    replacedItems: int = 0
    data_directory: str = "data"
    human_directory: str = "human"
    print("genisReplaceScores.py: Replacing selected_reviews.csv files in data directory with those from human directory.")
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1967, help="Seed value to specify the subdirectory.")
    args = parser.parse_args()

    for subdir in sorted(os.listdir(data_directory)):
        entryPath: str = os.path.join(data_directory, subdir)
        if not os.path.isdir(entryPath):
            # Skip files that may exist in the data directory (e.g., CSVs, pkl)
            continue
        reviewsFileName: str = os.path.join(entryPath, str(args.seed), 'selected_reviews.csv')
        if not os.path.exists(reviewsFileName):
            print(f"File {reviewsFileName} does not exist, skipping replacement.")
            continue
        humanSubdir: str = os.path.join(human_directory, subdir, str(args.seed))
        humanScoresFileName: str = os.path.join(humanSubdir, 'selected_reviews.csv')
        if not os.path.exists(humanScoresFileName):
            print(f"Human file {humanScoresFileName} does not exist, skipping replacement.")
            continue

        replacedItems = replaceHscoreInFiles(reviewsFileName, humanScoresFileName, "readable")
        print(f"Replaced {replacedItems} rows in {entryPath} from {humanScoresFileName}")

    print(f"All replacements done")
