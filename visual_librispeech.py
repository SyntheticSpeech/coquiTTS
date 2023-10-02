import os

# Define the path to your dataset directory
dataset_path = '/Users/apple/Downloads/LibriSpeech/dev-clean'

# Initialize a dictionary to store the count for each character
character_counts = {}

# Traverse the directory and subdirectories
for root, dirs, files in os.walk(dataset_path):
    # Split the path into components
    path_components = root.split(os.sep)
    
    # Check if the path is at least two levels deep (character/subdirectory)
    if len(path_components) >= 3:
        character = path_components[-2]  # The character is the second-to-last component
        file_count = len(files)  # Count the number of files in the current directory
        
        # Add the file count to the character's count
        if character in character_counts:
            character_counts[character] += file_count
        else:
            character_counts[character] = file_count

# Print the file counts for each character
for character, count in character_counts.items():
    print(f"Character '{character}': {count} files")
