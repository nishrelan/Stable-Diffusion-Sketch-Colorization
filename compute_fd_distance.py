from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import argparse
import os
import torch
from scipy import linalg
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--generations', type=str, required=True)
parser.add_argument('--targets', type=str, required=True)

def normalize(tensor):
    with torch.no_grad():
        return tensor / tensor.norm(p=2, dim=1, keepdim=True)

def get_images_from_directory(dir_name):
    images = []
    for filename in os.listdir(dir_name):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            print(filename)
            filepath = os.path.join(dir_name, filename)
            with open(filepath, 'rb') as f:
                 images.append(Image.open(f).convert("RGB").resize((224, 224)))

    return images

def get_mean_and_cov(tensor):

    means = torch.mean(tensor, dim=0)
    cov = torch.cov(tensor)

    return means, cov

def sqrtm(tensor, disp=True):
    m = tensor.detach().cpu().numpy().astype(np.float_)
    res, _ = linalg.sqrtm(m, disp=disp)
    sqrtm = torch.from_numpy(res.real).to(tensor)
    return sqrtm

def compute_fd(gen_stats, target_stats, eps=1e-6):
    with torch.no_grad():
        mu1, sigma1 = gen_stats
        mu2, sigma2 = target_stats

        diff = mu1 - mu2

        # Product might be almost singular
        covmean = sqrtm(sigma1@sigma2, disp=False)
        if not torch.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = torch.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if torch.is_complex(covmean):
            if not torch.allclose(torch.diagonal(covmean).imag, 0, atol=1e-3):
                m = torch.max(torch.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = torch.trace(covmean)

        return (torch.dot(diff, diff) + torch.trace(sigma1)
                + torch.trace(sigma2) - 2 * tr_covmean)

def main():
    args = parser.parse_args()

    print("Processing images...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # get the images from corresponding directories
    generation_images = get_images_from_directory(args.generations)
    target_images = get_images_from_directory(args.targets)

    generation_input = image_processor(images=generation_images, return_tensors='pt')
    target_input = image_processor(images=target_images, return_tensors='pt')

    generation_input = generation_input.to(device)
    target_input = target_input.to(device)

    gen_features = model.get_image_features(**generation_input)
    target_features = model.get_image_features(**target_input)

    print("Computing FD score")
    gen_stats = get_mean_and_cov(gen_features)
    target_stats = get_mean_and_cov(target_features)

    print(compute_fd(gen_stats, target_stats))


if __name__ == '__main__':
    main() 