import os

def rename_jfif_to_jpg(base_dir):
    print(f"Scanning {base_dir} for .jfif files...")
    count = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".jfif"):
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, file[:-5] + ".jpg")
                
                # Check if target exists
                if os.path.exists(new_path):
                    print(f"Skipping {file} -> Target .jpg already exists.")
                    continue
                
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {file} -> {os.path.basename(new_path)}")
                    count += 1
                except Exception as e:
                    print(f"Error renaming {file}: {e}")
    
    print(f"\nTotal files renamed: {count}")

if __name__ == "__main__":
    rename_jfif_to_jpg("dataset/Train")
