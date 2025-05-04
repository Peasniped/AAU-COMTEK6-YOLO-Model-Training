import os

# Set your folder path
folder_path = 'training\\images\\train'

# Get all files in the folder and filter to images only
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

# Sort images to have consistent ordering
images.sort()

# Rename each image
for i, filename in enumerate(images, start=1):
    new_name = f"horse{i:03d}.jpg"
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
    print(f"Renamed '{filename}' to '{new_name}'")

print("Renaming complete!")
