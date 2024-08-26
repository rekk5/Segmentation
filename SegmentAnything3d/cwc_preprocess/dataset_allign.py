import os
import re
from shutil import move
from collections import defaultdict

def extract_main_timestamp(filename):
    """Extract the main part of the timestamp from a filename (before the dot)."""
    match = re.match(r'(\d+)\.\d+', filename)
    return match.group(1) if match else None

def rename_files_in_pairs(base_path):
    directories = ['color', 'depth','depth_raw', 'depth_render', 'infrared', 'normal_render', 'clouds']

    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        file_groups = defaultdict(list)

        # Group files by the main timestamp
        for filename in os.listdir(dir_path):
            main_timestamp = extract_main_timestamp(filename)
            if main_timestamp:
                file_groups[main_timestamp].append(filename)

        # Rename files within each group
        for main_timestamp, filenames in file_groups.items():
            filenames.sort()  # Sort to maintain consistency in ordering
            for idx, filename in enumerate(filenames):
                old_file_path = os.path.join(dir_path, filename)
                new_file_path = os.path.join(dir_path, f"{main_timestamp}.{idx}{os.path.splitext(filename)[1]}")

                if os.path.exists(old_file_path):
                    print(f"Renaming {old_file_path} to {new_file_path}")
                    move(old_file_path, new_file_path)
                else:
                    print(f"File {old_file_path} does not exist, skipping...")

            # If there's only one file, ensure it's renamed to .0
            if len(filenames) == 1:
                single_file = filenames[0]
                old_file_path = os.path.join(dir_path, single_file)
                new_file_path = os.path.join(dir_path, f"{main_timestamp}.0{os.path.splitext(single_file)[1]}")
                if old_file_path != new_file_path:  # Only rename if the file isn't already correctly named
                    if os.path.exists(old_file_path):
                        print(f"Renaming {old_file_path} to {new_file_path}")
                        move(old_file_path, new_file_path)
                    else:
                        print(f"File {old_file_path} does not exist, skipping...")

if __name__ == "__main__":
    base_dataset_path = '/home/t/Desktop/work/datasets/cwc'
    rename_files_in_pairs(base_dataset_path)
