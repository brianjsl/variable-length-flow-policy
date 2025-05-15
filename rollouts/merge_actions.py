import pickle
import numpy as np
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Merge .pkl files by timestep and batch.')
parser.add_argument('folder_path', type=str, help='Path to the folder containing .pkl files')

# Parse the arguments
args = parser.parse_args()

# Directory path where the .pkl files are located
folder_path = args.folder_path

# List to store all the actions from the pkl files
all_actions_by_timestep = []

# Check if the folder path exists
if not os.path.exists(folder_path):
    print(f"Error: The directory {folder_path} does not exist.")
    exit(1)

print_shape = True
# Iterate through the pkl files and load the data
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith('.pkl'):
        file_path = os.path.join(folder_path, filename)

        # Load the pkl file
        with open(file_path, 'rb') as f:
            actions = pickle.load(f)  # Assuming actions is a list of (28, 1, 10) shaped arrays

        # Convert list to a numpy array of shape (num_timesteps, 1, 10)

        actions = np.array(actions).squeeze()  # Shape should be (500, 28, 1, 10)
        # B, T, N, D = actions.shape  # B=batch, T=time, N=sub-batch, D=dim
        # actions = actions.transpose(0, 2, 1, 3).reshape(B * N, T, D)
        if print_shape:
            print(actions.shape)
            print_shape = False

        # Reshape the actions to (28, 500, 10)
        actions_by_timestep = actions.transpose(1, 0, 2)  # Transpose to shape (28, 500, 10)

        # Append the actions by timestep to the list
        all_actions_by_timestep.append(actions_by_timestep)

# Now concatenate them by batch
merged_actions = np.concatenate(all_actions_by_timestep, axis=0)  # Shape: (46, 500, 10)

# Define the path for the merged actions file in the same folder
output_file_path = os.path.join(folder_path, 'merged_actions.pkl')

# Save the result to merged_actions.pkl
with open(output_file_path, 'wb') as f:
    pickle.dump(merged_actions, f)

print(f"Merging complete. The result has been saved as '{output_file_path}'.")