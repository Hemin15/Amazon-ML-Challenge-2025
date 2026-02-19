import pandas as pd
from src.utils import download_images

if __name__ == "__main__":
    # Load only the training CSV
    train = pd.read_csv('dataset/raw/train.csv')

    # Folder to save images
    download_folder = 'dataset/images/train/'

    # Extract only unique, valid image links
    all_image_links = train['image_link'].dropna().unique().tolist()

    print(f"Total train images to download: {len(all_image_links)}")

    # Download them
    download_images(all_image_links, download_folder)
