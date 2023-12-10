from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import argparse
import os
import torch

parser = argparse.ArgumentParser()

parser.add_argument('-d', type=str, required=True)

def normalize(tensor):
    with torch.no_grad():
        return tensor / tensor.norm(p=2, dim=1, keepdim=True)

def main():
    args = parser.parse_args()
    directory = args.d

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")




    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            print(filename)
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as f:
                 images.append(Image.open(f).convert("RGB").resize((224, 224)))


    # get images as tensors and then run inference on the clip image
    # model to get image embeddings
    im_input = image_processor(images=images, return_tensors='pt')
    im_input = im_input.to(device)
    im_features = model.get_image_features(**im_input)


    prompt2 = "a black and white image"
    prompt3 = "an image with color"
    prompt6 = "a photo of a pokemon"
    prompt7 = "a colored photo of a pokemon"

    text_inputs = tokenizer([prompt2, prompt3, prompt6, prompt7], padding=True, return_tensors='pt')
    text_inputs = text_inputs.to(device)
    text_features = model.get_text_features(**text_inputs)

    # compute cosine similarities between the prompt embedding and each image
    im_features = normalize(im_features)
    text_features = normalize(text_features)

    similarities = im_features @ text_features.T

    print(similarities)

    


if __name__ == '__main__':
    main()