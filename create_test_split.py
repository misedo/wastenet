# create_test_split.py
import os
import shutil
import random

# --- CONFIGURATION ---
# 1. Set the path to your main 'train' directory.
#    (Use the full path if you are unsure)
SOURCE_TRAIN_DIR = 'wastedata_split/train' 

# 2. Set the path for the new 'test' directory that will be created.
TEST_DIR = 'wastedata_split/test'

# 3. Set the fraction of files you want to move from train to test (e.g., 0.2 for 20%).
TEST_SPLIT_FRACTION = 0.20
# --- END OF CONFIGURATION ---


def create_test_split():
    """
    Identifies class subdirectories in a source training directory, creates a parallel
    test directory, and moves a random fraction of files from each class to the
    new test directory.
    """
    print("Starting to create the test split...")
    print(f"Source: {os.path.abspath(SOURCE_TRAIN_DIR)}")
    print(f"Destination: {os.path.abspath(TEST_DIR)}")

    if not os.path.exists(SOURCE_TRAIN_DIR):
        print(f"Error: Source training directory not found at '{SOURCE_TRAIN_DIR}'")
        return

    # Get all class subdirectories (e.g., 'paper', 'plastic', etc.)
    try:
        class_dirs = [d for d in os.listdir(SOURCE_TRAIN_DIR) if os.path.isdir(os.path.join(SOURCE_TRAIN_DIR, d))]
        if not class_dirs:
            print(f"Error: No class subdirectories found in '{SOURCE_TRAIN_DIR}'.")
            print("Please ensure your train directory is structured like: train/class1, train/class2, ...")
            return
    except FileNotFoundError:
        print(f"Error: Could not access the directory '{SOURCE_TRAIN_DIR}'. Please check the path.")
        return

    # Create the main test directory
    os.makedirs(TEST_DIR, exist_ok=True)
    
    total_files_moved = 0

    # Loop through each class directory
    for class_name in class_dirs:
        source_class_path = os.path.join(SOURCE_TRAIN_DIR, class_name)
        dest_class_path = os.path.join(TEST_DIR, class_name)
        
        # Create corresponding class directory in the test folder
        os.makedirs(dest_class_path, exist_ok=True)
        
        # Get all image files for the class
        images = [f for f in os.listdir(source_class_path) if os.path.isfile(os.path.join(source_class_path, f))]
        
        # Shuffle the list of images to ensure the selection is random
        random.shuffle(images)
        
        # Calculate the number of files to move
        num_test_files = int(len(images) * TEST_SPLIT_FRACTION)
        
        # Select the files to move
        files_to_move = images[:num_test_files]
        
        if not files_to_move:
            print(f"Warning: No files selected to move for class '{class_name}'. The class might be too small.")
            continue
            
        print(f"\nProcessing class: {class_name}")
        print(f"Moving {len(files_to_move)} of {len(images)} files...")

        # Move each selected file
        for file_name in files_to_move:
            source_file = os.path.join(source_class_path, file_name)
            dest_file = os.path.join(dest_class_path, file_name)
            shutil.move(source_file, dest_file)
        
        total_files_moved += len(files_to_move)

    print(f"\n-------------------------------------------------")
    print(f"Test split creation complete!")
    print(f"Total files moved: {total_files_moved}")
    print(f"A new 'test' directory has been created at: {os.path.abspath(TEST_DIR)}")
    print(f"-------------------------------------------------")


if __name__ == "__main__":
    create_test_split()
    