from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import argparse
import os
import torch

parser = argparse.ArgumentParser()

parser.add_argument('-d', type=str, required=True)

def normalize(tensor):
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
        if filename.endswith('.png') or filename.endswith('.jpg'):
            print(filename)
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as f:
                 images.append(Image.open(f).convert("RGB").resize((224, 224)))


    # get images as tensors and then run inference on the clip image
    # model to get image embeddings
    im_input = image_processor(images=images, return_tensors='pt')
    with torch.no_grad():
        im_input = im_input.to(device)
        im_features = model.get_image_features(**im_input)

        prompt = "a colored photo of a pokemon"
        prompt2 = "test prompt" # testing to see if this pipeline works

        text_inputs = tokenizer([prompt, prompt2], padding=True, return_tensors='pt')
        text_inputs = text_inputs.to(device)
        text_features = model.get_text_features(**text_inputs)

        # compute cosine similarities between the prompt embedding and each image
        im_features = normalize(im_features)
        text_features = normalize(text_features)

        similarities = im_features @ text_features.T

        mean_similarity = similarities.mean(dim=0)

        print(100*(1 - mean_similarity))

    


    


if __name__ == '__main__':
    main()