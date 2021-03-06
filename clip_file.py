
import numpy as np
import torch
import os
import clip
from PIL import Image
import glob
from PIL import Image
def find_class(image,classes):
    print("hey waqas")
    if(len(classes) != 0):
        file = open('_tokenization.txt' , 'a')
        file.write(classes)
        file.close()
    candidate_captions = []
    with open('_tokenization.txt') as f:
        candidate_captions = f.read().splitlines()
    class_names = candidate_captions
    result = process(image,candidate_captions)
    print("waqas anwar")
    return result


def process (img, candidate_captions):    
    print("Muhammad Waqas")
    def argmax(iterable):
        return max(enumerate(iterable), key=lambda x: x[1])[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("ViT-B/32", device=device)

    correct = []

    #define our target classificaitons, you can should experiment with these strings of text as you see fit, though, make sure they are in the same order as your class names above
    text = clip.tokenize(candidate_captions).to(device)

    class_correct = []
    image = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
   
        pred = candidate_captions[argmax(list(probs)[0])]
        print(pred)
        return pred
