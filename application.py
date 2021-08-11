from clip_file import find_class
import cv2
import numpy as np
from PIL import Image
from flask import Flask,request,render_template

application = Flask(__name__)

@application.route('/')
def home():
    return render_template('front1.html')
@application.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        image_file = request.files['file']
        class_text = request.form['classname']
        image_file = np.fromfile(image_file, np.uint8)
        image_file = cv2.imdecode(image_file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        result_class = find_class(im_pil,class_text)
        return render_template("result.html",result_class = result_class)

if __name__ =='__main__':
    application.run(debug=True)
    application.config['TEMPLATES_AUTO_RELOAD'] = True
