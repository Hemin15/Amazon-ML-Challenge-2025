import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPModel
from tqdm import tqdm

# CONFIG
IMAGE_DIR = r"D:\Amazon\dataset\images\train_mapped"
PRELOAD_DIR = r"D:\Amazon\dataset\preloaded_images"  # will save .pt tensors here
OUTPUT_FILE = r"D:\Amazon\output\clip_embeddings.npy"
BATCH_SIZE = 32
USE_AMP = True
MODEL_NAME = "openai/clip-vit-large-patch14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

# PRELOAD IMAGES AS .PT
def preload_images():
    os.makedirs(PRELOAD_DIR, exist_ok=True)
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"Preloading {len(files)} images...")
    for f in tqdm(files, desc="Preloading images"):
        img_path = os.path.join(IMAGE_DIR, f)
        try:
            from PIL import Image, UnidentifiedImageError
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError):
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        tensor = transform(image)
        torch.save(tensor, os.path.join(PRELOAD_DIR, f"{os.path.splitext(f)[0]}.pt"))

# DATASET
class PreloadedDataset(Dataset):
    def __init__(self, preload_dir):
        self.files = [f for f in os.listdir(preload_dir) if f.endswith(".pt")]
        self.dir = preload_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        tensor = torch.load(os.path.join(self.dir, f))
        return tensor, f.replace(".pt", "")

# CUSTOM COLLATE
def collate_fn(batch):
    tensors, names = zip(*batch)
    return torch.stack(tensors), list(names)

# MAIN FUNCTION
def extract_clip_embeddings():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load model
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    dataset = PreloadedDataset(PRELOAD_DIR)
    total_images = len(dataset)
    print(f"Total preloaded images: {total_images}")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0,
                            pin_memory=True, collate_fn=collate_fn)

    embedding_dim = model.config.projection_dim
    embeddings = np.zeros((total_images, embedding_dim), dtype=np.float32)

    idx = 0
    for batch_imgs, batch_names in tqdm(dataloader, desc="CLIP Embedding Extraction", dynamic_ncols=True):
        batch_imgs = batch_imgs.to(DEVICE)

        with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
            with torch.no_grad():
                img_features = model.get_image_features(batch_imgs)
                img_features /= img_features.norm(dim=-1, keepdim=True)
                embeddings[idx:idx+batch_imgs.size(0), :] = img_features.cpu().numpy()
        idx += batch_imgs.size(0)

    np.save(OUTPUT_FILE, embeddings)
    print(f"Saved embeddings to {OUTPUT_FILE}")

if __name__ == "__main__":
    # Run once to preload images
    if not os.path.exists(PRELOAD_DIR) or len(os.listdir(PRELOAD_DIR)) == 0:
        preload_images()
    extract_clip_embeddings()
