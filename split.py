# split_data.py
import splitfolders

# The path to your original dataset with 10 class sub-folders
input_folder = '/workspace/project/wastedata'

# The path where the new 'train' and 'val' folders will be created
output_folder = '/workspace/project/wastedata_split'

# Split with a ratio.
# .8 means 80% for training, .2 means 20% for validation.
print(f"Splitting folder {input_folder} into {output_folder}...")
splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.8, .2))
print("Done.")
