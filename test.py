#from clip_file import find_class
import cv2
import numpy as np
from PIL import Image
from flask import Flask,request,render_template
import torch
import os
import clip
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
models = clip.available_models()
print(models)
model, transform = clip.load("RN101", device=device)


application = Flask(__name__)

@application.route('/')
def home():
    return render_template('front1.html')
@application.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        image_file = request.files['file']
        class_text = request.form['classname']
#        print("waqas anwar")
        image_file = np.fromfile(image_file, np.uint8)
        image_file = cv2.imdecode(image_file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        result_class = find_class(im_pil,class_text)
 #       print(result_class)
        return render_template("result.html",result_class = result_class)


def find_class(image,classes):
    
    if(len(classes) != 0):
        file = open('_tokenization.txt' , 'a')
        file.write(classes)
        file.close()
    candidate_captions = []
    with open('_tokenization.txt') as f:
        candidate_captions = f.read().splitlines()
    class_names = candidate_captions
    result = process(image,candidate_captions)
    #print("Muhammad Waqas")
    return result


def process (img, candidate_captions): 
    print("hey waqas")   
    def argmax(iterable):
        return max(enumerate(iterable), key=lambda x: x[1])[0]

    correct = []

    #define our target classificaitons, you can should experiment with these strings of text as you see fit, though, make sure they are in the same order as your class names above
    text = clip.tokenize(candidate_captions).to(device)
    print("clip testing3")

    class_correct = []
    print("clip testing4")
    image = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("Muhammad Waqas")

        pred = candidate_captions[argmax(list(probs)[0])]
        print(pred)
        return pred



if __name__ =='__main__':
    application.run(debug=True,host='0.0.0.0', port=5000)
    application.config['TEMPLATES_AUTO_RELOAD'] = True
