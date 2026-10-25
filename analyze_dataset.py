import os
import hashlib
from PIL import Image
from collections import defaultdict

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def analyze_dataset(base_dir):
    print(f"Analyzing dataset at: {base_dir}")
    
    if not os.path.exists(base_dir):
        print("Dataset directory not found.")
        return

    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    total_stats = {}

    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        files = os.listdir(class_dir)
        
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.jfif'))]
        
        hashes = defaultdict(list)
        valid_images = 0
        corrupt_images = 0
        small_images = 0
        
        print(f"\nScanning class: {class_name}")
        print(f"  Total files: {len(files)}")
        print(f"  Image files: {len(image_files)}")

        for img_file in image_files:
            file_path = os.path.join(class_dir, img_file)
            
            # Check for corruption and size
            try:
                with Image.open(file_path) as img:
                    img.verify() # invalidates img
                
                # Reopen to check size (verify closes it)
                with Image.open(file_path) as img:
                    width, height = img.size
                    if width < 50 or height < 50:
                        small_images += 1
                
                valid_images += 1
                
                # Check hash for uniqueness
                file_hash = calculate_md5(file_path)
                hashes[file_hash].append(img_file)

            except Exception as e:
                # print(f"    Corrupt/Invalid: {img_file} - {str(e)}")
                corrupt_images += 1

        unique_images = len(hashes)
        duplicates = valid_images - unique_images
        
        print(f"  Valid Images: {valid_images}")
        print(f"  Corrupt Images: {corrupt_images}")
        print(f"  Small Images (<50px): {small_images}")
        print(f"  Unique Images (by content): {unique_images}")
        print(f"  Duplicates: {duplicates}")
        
        if unique_images > 0:
            dupe_ratio = duplicates / valid_images
            if dupe_ratio > 0.5:
                print(f"  [WARNING] High duplication ratio: {dupe_ratio:.2f}")

        total_stats[class_name] = {
            "total": len(image_files),
            "unique": unique_images,
            "corrupt": corrupt_images
        }

    with open("dataset_analysis.txt", "w") as f:
        f.write("="*40 + "\n")
        f.write("SUMMARY\n")
        f.write("="*40 + "\n")
        f.write(f"{'Class Name':<40} | {'Total':<10} | {'Unique':<10} | {'Corrupt':<10}\n")
        f.write("-" * 80 + "\n")
        for cls, stats in total_stats.items():
            f.write(f"{cls:<40} | {stats['total']:<10} | {stats['unique']:<10} | {stats['corrupt']:<10}\n")
    
    print("Analysis complete. Saved to dataset_analysis.txt")

if __name__ == "__main__":
    analyze_dataset("dataset/Train")
