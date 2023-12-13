import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image
import requests
from PIL import Image
import argparse
import os
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)

def get_images_from_directory(dir_name, resize_to):
    images = []
    filenames = []
    for filename in os.listdir(dir_name):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            filenames.append(filename)
            filepath = os.path.join(dir_name, filename)
            with open(filepath, 'rb') as f:
                 images.append(Image.open(f).convert("RGB").resize(resize_to))

    return images, filenames

def save_images_to_directory(dir_name, images, filenames):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for im, filename in zip(images, filenames):
        img_path = os.path.join(dir_name, filename)
        im.save(img_path)

def main():
    args = parser.parse_args()

    device = "cuda:0"
    model_id_or_path = "kandinsky-community/kandinsky-2-2-decoder"
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id_or_path, torch_dtype=torch.float16, use_safetensors=True
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()

    images, filenames = get_images_from_directory(args.input_dir, resize_to=(768, 512))


    prompt = "A colorized version of the pokemon sketch."
    prompt = "A colorized version of the specific pokemon type in the sketch."
    prompt = "A colorized version of the image"
    prompt = "in color"

    output_images = []
    output_filenames = []
    for img, filename in zip(images, filenames):
        output_im = pipe(prompt=prompt, image=img, strength=0.2, guidance_scale=20, num_images_per_prompt=1).images[0]
        output_images.append(output_im)
        comps = filename.split('.')
        output_filenames.append(''.join([comps[0], '_gen.', comps[1]]))


    save_images_to_directory(args.output_dir, output_images, output_filenames)

if __name__ == '__main__':
    main()