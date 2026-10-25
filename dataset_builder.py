import os
import shutil
from bing_image_downloader import downloader

def create_dataset():
    # Define the output directory
    output_dir = "dataset"
    
    # Define the specific spots and their download limits
    spots_config = {
        "Raigad Fort Maha Darwaja": 150, # Increased limit as requested
        "Raigad Fort Takmak Tok": 50,
        "Raigad Fort Jagadishwar Temple": 50,
        "Raigad Fort Shivaji Maharaj Statue": 50,
        "Raigad Fort Hirkani Buruj": 50
    }
    
    print(f"Starting download for {len(spots_config)} classes...")
    
    for spot, limit in spots_config.items():
        print(f"\nDownloading {limit} images for: {spot}")
        
        # The downloader creates a subfolder matching the 'query_string' inside 'output_dir' -- actually it uses the query text
        downloader.download(
            query=spot,
            limit=limit,
            output_dir=output_dir,
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
            verbose=True
        )
        
    print("\nDownload complete!")
    print(f"Images are saved in the '{output_dir}' directory.")
    print("IMPORTANT: Please manually review the folders and delete any irrelevant images before training.")

if __name__ == "__main__":
    create_dataset()
