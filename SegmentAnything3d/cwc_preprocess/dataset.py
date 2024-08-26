import os
import re
from collections import defaultdict

def extract_main_timestamp(line):
    """Extract the main part of the timestamp from a pose file line (before the dot)."""
    parts = line.strip().split()
    timestamp = parts[0]
    match = re.match(r'(\d+)\.(\d+)', timestamp)
    return match.group(1) if match else None

def rename_timestamps_in_pose_file(pose_file_path):
    with open(pose_file_path, 'r') as f:
        lines = f.readlines()

    timestamp_groups = defaultdict(list)

    # Group lines by main timestamp
    for line in lines:
        main_timestamp = extract_main_timestamp(line)
        if main_timestamp:
            timestamp_groups[main_timestamp].append(line)

    # Create new lines with updated timestamps
    new_lines = []
    for main_timestamp, group in timestamp_groups.items():
        if len(group) == 1:
            # Single timestamp, append .0
            parts = group[0].strip().split()
            new_timestamp = f"{main_timestamp}.0"
            new_line = f"{new_timestamp} {' '.join(parts[1:])}\n"
            new_lines.append(new_line)
        elif len(group) == 2:
            # Two timestamps, append .0 and .1
            for idx, line in enumerate(group):
                parts = line.strip().split()
                new_timestamp = f"{main_timestamp}.{idx}"
                new_line = f"{new_timestamp} {' '.join(parts[1:])}\n"
                new_lines.append(new_line)
        else:
            # More than two timestamps, handle accordingly (not expected based on description)
            for idx, line in enumerate(group):
                parts = line.strip().split()
                new_timestamp = f"{main_timestamp}.{idx}"
                new_line = f"{new_timestamp} {' '.join(parts[1:])}\n"
                new_lines.append(new_line)

    # Write the updated lines back to the pose file
    with open(pose_file_path, 'w') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    base_dataset_path = '/home/t/Desktop/work/datasets/cwc/poses'

    # Update the poses_color.txt file
    poses_color_path = os.path.join(base_dataset_path, 'poses_color.txt')
    rename_timestamps_in_pose_file(poses_color_path)

    # Update the poses_infrared.txt file if needed
    poses_infrared_path = os.path.join(base_dataset_path, 'poses_infrared.txt')
    rename_timestamps_in_pose_file(poses_infrared_path)
