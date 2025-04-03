import os

def create_txt_files_from_jpg(jpg_dir, source_txt_path, output_txt_dir):
    """
    Creates a copy of the source TXT file for every JPG file in the directory.

    Args:
        jpg_dir (str): Path to the directory containing JPG files.
        source_txt_path (str): Path to the source TXT file to be copied.
        output_txt_dir (str): Directory where the new TXT files will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_txt_dir, exist_ok=True)

    # Read the source TXT content once
    with open(source_txt_path, 'r') as f:
        txt_content = f.read()

    # Iterate over all JPG files in the directory
    for jpg_file in os.listdir(jpg_dir):
        if jpg_file.lower().endswith('.jpg'):
            # Get the base name (without extension)
            base_name = os.path.splitext(jpg_file)[0]

            # Define the output TXT file path
            output_txt_path = os.path.join(output_txt_dir, f"{base_name}.txt")

            # Write the same content to the new TXT file
            with open(output_txt_path, 'w') as f:
                f.write(txt_content)

    print(f"Created {len([f for f in os.listdir(jpg_dir) if f.endswith('.jpg')])} TXT files in {output_txt_dir}")

# Example usage:
jpg_directory = "/home/lore_be/data/vkitti2/kitti_format/image_2/"  # Directory containing JPG files
source_txt_file = "/home/lore_be/data/vkitti2/kitti_format/calib/000000.txt"  # Source TXT file to copy
output_directory = "/home/lore_be/data/vkitti2/kitti_format/calib/"  # Where to save the new TXT files

create_txt_files_from_jpg(jpg_directory, source_txt_file, output_directory)