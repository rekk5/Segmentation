import os

# Define the paths
root_dir = '/home/t/Desktop/work/datasets/cwc'  # Root directory of your dataset
folders_to_clean = ['clouds', 'color', 'depth', 'infrared']
pose_file_color = os.path.join(root_dir, 'poses', 'poses_color.txt')
pose_file_infrared = os.path.join(root_dir, 'poses', 'poses_infrared.txt')

def delete_files_with_suffix(directory, suffix):
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            os.remove(os.path.join(directory, filename))
            print(f"Deleted {filename} from {directory}")

def clean_pose_file(pose_file, suffix='.1'):
    with open(pose_file, 'r') as file:
        lines = file.readlines()
    with open(pose_file, 'w') as file:
        for line in lines:
            if not line.startswith(suffix):
                file.write(line)
            else:
                print(f"Removed line from {pose_file}: {line.strip()}")

# Delete .1 files from the specified directories
for folder in folders_to_clean:
    delete_files_with_suffix(os.path.join(root_dir, folder), '.1.jpg')
    delete_files_with_suffix(os.path.join(root_dir, folder), '.1.png')
    delete_files_with_suffix(os.path.join(root_dir, folder), '.1.ply')

# Clean the poses files
clean_pose_file(pose_file_color, suffix='1000000.1')
clean_pose_file(pose_file_infrared, suffix='1000000.1')

print("Clean-up completed!")
