import os
import sys

import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, '/home/dominic/f3rm')

from f3rm.features import clip
from f3rm.features.clip import tokenize
from f3rm.features.clip_extract import CLIPArgs, extract_clip_features

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGE_DIR = os.path.join(_MODULE_DIR, "images")

min_sim = 0.05
max_sim = 0.35

#image_paths = ['/home/dominic/concept-graphs/Datasets/office_0/Sequence_1/rgb/rgb_633.png', '/home/dominic/#concept-graphs/Datasets/office_0/Sequence_1/rgb/rgb_634.png']

image_paths = ['/home/dominic/cam_ws/imgs/2024-01-14-11-29-29/images/rgb_523.jpg']

@torch.no_grad()
def demo_clip_features(text_query: str) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Extract the patch-level features for the images
    clip_embs = extract_clip_features(image_paths, device)
    clip_embs /= clip_embs.norm(dim=-1, keepdim=True)

    # Load the CLIP model so we can get text embeddings
    model, _ = clip.load(CLIPArgs.model_name, device=device)

    # Encode text query
    tokens = tokenize(text_query).to(device)
    text_embs = model.encode_text(tokens)
    text_embs /= text_embs.norm(dim=-1, keepdim=True)

    # Compute similarities
    sims = clip_embs @ text_embs.T
    sims_first = sims[0,:,:,:]
    print(clip_embs.shape)
    print(sims.shape)
    indices = torch.nonzero(sims_first > 0.20)
    print(indices)
    
    print(clip_embs[0, indices[:, 0], indices[:, 1],:].shape)
    avg = torch.mean(clip_embs[0, indices[:, 0], indices[:, 1],:], dim=0)
    print('avg shape', avg.shape)
    clip_embs[0, indices[:, 0], indices[:, 1],:] = avg
    clip_embs /= clip_embs.norm(dim=-1, keepdim=True)
    sims = clip_embs @ text_embs.T
    
    
    #print(q)
    sims = sims.squeeze()

    # Visualize
    plt.figure()
    cmap = plt.get_cmap("turbo")
    for idx, (image_path, sim) in enumerate(zip(image_paths, sims)):
        sim = sims
        print(sims.shape)
        plt.subplot(2, len(image_paths), idx + 1)
        plt.imshow(Image.open(image_path))
        plt.title(os.path.basename(image_path))
        plt.axis("off")

        plt.subplot(2, len(image_paths), len(image_paths) + idx + 1)
        print(sim.min())
        print(sim.max())
        #sim[indices[:, 0], indices[:, 1]] = 0.5
        sim_norm = (sim - min_sim) / (max_sim - min_sim)
        heatmap = cmap(sim_norm.cpu().numpy())
        plt.imshow(heatmap)
        plt.axis("off")

    plt.tight_layout()
    plt.suptitle(f'Similarity to language query "{text_query}"')

    text_label = text_query.replace(" ", "-")
    plt_fname = f"demo_clip_features_{text_label}.png"
    #plt.savefig(plt_fname)
    #print(f"Saved plot to {plt_fname}")
    plt.show()


if __name__ == "__main__":
    demo_clip_features(text_query="pile of chocolate snacks")
    
    
    
    
    

