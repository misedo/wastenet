# find_bad_images.py
import os
from PIL import Image
import tensorflow as tf

# --- Configuration ---
# IMPORTANT: Update this path to your new 10-class dataset
ROOT_DATA_DIR = '/workspace/project/wastedata'
# Set to True to automatically delete bad files, False to only list them.
# START WITH 'False' TO BE SAFE!
AUTO_DELETE_BAD_FILES = False
# =========================================================


def find_corrupted_images(root_dir):
    """
    Recursively finds and lists corrupted or non-image files in a directory.
    """
    bad_files = []
    total_files_checked = 0
    print(f"Starting verification in directory: {root_dir}")

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                total_files_checked += 1
                file_path = os.path.join(dirpath, filename)
                try:
                    # Use TensorFlow's own I/O to be 100% sure
                    # This is more robust than Pillow for this specific error
                    image_bytes = tf.io.read_file(file_path)
                    tf.io.decode_image(image_bytes, channels=3)

                    # Optional: A secondary check with Pillow
                    # img = Image.open(file_path)
                    # img.verify()

                except Exception as e:
                    print(f"❌ Problem file found: {file_path}")
                    print(f"   Reason: {e}")
                    bad_files.append(file_path)

    print("\n" + "="*50)
    print("Verification Complete.")
    print(f"Total image files checked: {total_files_checked}")
    return bad_files

if __name__ == '__main__':
    corrupted_list = find_corrupted_images(ROOT_DATA_DIR)
    
    print(f"Found {len(corrupted_list)} problematic files.")
    
    if corrupted_list:
        print("\nList of problematic files:")
        for f in corrupted_list:
            print(f)
            
        if AUTO_DELETE_BAD_FILES:
            print("\nAUTO_DELETE_BAD_FILES is set to True. Deleting files...")
            for f in corrupted_list:
                try:
                    os.remove(f)
                    print(f"Deleted: {f}")
                except Exception as e:
                    print(f"Could not delete {f}. Reason: {e}")
            print("Deletion complete.")
        else:
            print("\nTo automatically delete these files, set AUTO_DELETE_BAD_FILES = True in the script.")
    else:
        print("\n✅ No problematic files found. Your dataset looks clean!")