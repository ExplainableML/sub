import os
import torch
import argparse
import open_clip
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pickle
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, preprocess, file_path):
        self.preprocess = preprocess
        column_names = ['file', 'changed_attr', 'bird']
        img_paths = pd.read_csv(file_path, header=None)
        img_paths.columns = column_names
        self.images = img_paths
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_data = self.images.iloc[index]
        img_path = img_data['file']
        image = Image.open(img_path)
        image = self.preprocess(image)
        return image
    
    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--file_path", type=str, required=True)
    args = parser.parse_args()

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    # Move model to GPU
    model.to("cuda")
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    
    # Load the attribute queries
    attribute_queries = pd.read_csv("attribute_clip_queries.csv")
    # Embed the attribute queries
    attribute_queries_embeddings = model.encode_text(tokenizer(attribute_queries["value"].tolist()).cuda())
    attribute_queries_embeddings /= attribute_queries_embeddings.norm(dim=-1, keepdim=True)
    attribute_queries_embeddings = attribute_queries_embeddings.cuda()

    # Load dataset
    dataset = ImageDataset(preprocess, args.file_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ImageDataset.collate_fn
    )
    
    results_df = pd.DataFrame(index=range(len(dataset)), columns=["image_path"] + attribute_queries["attribute"].tolist())
    running_index = 0

    # Run inference
    for images in tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            image_features = model.encode_image(images.cuda())
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarities between image features and attribute queries
            similarities = image_features @ attribute_queries_embeddings.T
            similarities = similarities.cpu().numpy()
            
            for row in similarities:
                image_path = dataset.images.iloc[running_index]['file']
                results_df.iloc[running_index] = [image_path] + row.tolist()
                running_index += 1
    
    os.makedirs(f"results/menon_vondrick_clip/{args.model}_{args.pretrained}", exist_ok=True)
    results_df.to_csv(f"results/menon_vondrick_clip/{args.model}_{args.pretrained}/attribute_clip_our_results.csv", index=False)
