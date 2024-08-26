import os

# Define the paths
root_dir = '/home/t/Desktop/work/datasets/cwc'  # Change this to the root directory of your dataset
folders_to_clean = ['clouds', 'color', 'depth', 'infrared']
pose_file_color = os.path.join(root_dir, 'poses', 'poses_color.txt')
pose_file_infrared = os.path.join(root_dir, 'poses', 'poses_infrared.txt')

def delete_every_other_file(directory):
    files = sorted(os.listdir(directory))  # Sort to ensure the order is correct
    for i, filename in enumerate(files):
        if i % 2 != 0:  # Delete every other file (keep if index is even)
            os.remove(os.path.join(directory, filename))
            print(f"Deleted {filename} from {directory}")

def clean_pose_file(pose_file):
    with open(pose_file, 'r') as file:
        lines = file.readlines()
    with open(pose_file, 'w') as file:
        for i, line in enumerate(lines):
            if i % 2 == 0:  # Keep every other line
                file.write(line)
            else:
                print(f"Removed line from {pose_file}: {line.strip()}")

# Delete every other file from the specified directories
for folder in folders_to_clean:
    delete_every_other_file(os.path.join(root_dir, folder))

# Clean the poses files
clean_pose_file(pose_file_color)
clean_pose_file(pose_file_infrared)

print("Clean-up completed!")
