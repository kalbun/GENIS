# Directory data contains various subdirectories, and each of them has
# a subdirectory named after the seed value (for example, data/xxx/1967)
# The directory "human" shares the same structure.
# For each subdirectory, replace the file named "selected_reviews.csv" in "data"
# # with the file named "selected_reviews.csv" in "human".
import os
import shutil
import argparse

def replace_selected_reviews(data_dir: str, human_dir: str, seed: int) -> None:
    """
    Replace the 'selected_reviews.csv' file in each subdirectory of 'data_dir'
    with the one from the corresponding subdirectory in 'human_dir'.

    Args:
        data_dir (str): The path to the data directory containing subdirectories.
        human_dir (str): The path to the human directory containing subdirectories.
        seed (int): The seed value to specify the subdirectory.
    """
    for subdir in os.listdir(data_dir):
        fileToReplace = os.path.join(data_dir, subdir, str(seed), 'selected_reviews.csv')
        if not os.path.exists(fileToReplace):
            print(f"File {fileToReplace} does not exist, skipping replacement.")
            continue
        humanSubdir = os.path.join(human_dir, subdir, str(seed))
        humanFile = os.path.join(humanSubdir, 'selected_reviews.csv')
        if not os.path.exists(humanFile):
            print(f"Human file {humanFile} does not exist, skipping replacement.")
            continue
        # Replace the file
        shutil.copy(humanFile, fileToReplace)
        print(f"Replaced {fileToReplace} with {humanFile}")

if __name__ == "__main__":

    data_directory: str = "data"
    human_directory: str = "human"
    print("genisReplaceScores.py: Replacing selected_reviews.csv files in data directory with those from human directory.")
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1967, help="Seed value to specify the subdirectory.")
    args = parser.parse_args()
    replace_selected_reviews(data_directory, human_directory, args.seed)
    print("All replacements done.")
