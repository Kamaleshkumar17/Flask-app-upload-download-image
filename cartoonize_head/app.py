from flask import Flask, render_template, request, redirect, url_for,send_file
from werkzeug.utils import secure_filename
import cv2
#from sklearn.cluster import KMeans
from imutils import face_utils
import numpy as np
import os
#import matplotlib.pyplot as plt
import dlib
import numpy as np
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
ANIMATED_FOLDER='animated/'
app.template_folder = 'templates'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANIMATED_FOLDER'] = ANIMATED_FOLDER
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('no filename')
            return redirect(request.url)
        else:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("saved file successfully")
            #send file name as parameter to downlad
            return redirect('/process-file/'+ filename)
    return render_template('up.html')
@app.route('/process-file/<filename>', methods = ['GET'])
def process_file(filename):
    file_path = UPLOAD_FOLDER + filename
    dirName = 'animated/'
    try:
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    fname, extension = os.path.splitext(filename)
    processed_file_name = dirName + fname +'_processed' + extension
    img = cv2.imread(file_path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"F:\CV\shape_predictor_68_face_landmarks.dat")
    faces = detector(img, 1)
    for i, face in enumerate(faces):
        landmarks = predictor(img, face)
        x = [landmarks.part(n).x for n in range(68)]
        y = [landmarks.part(n).y for n in range(68)]
        x1 = min(x) - int(0.2 * (max(x) - min(x)))
        y1 = min(y) - int(0.3 * (max(y) - min(y)))
        x2 = max(x) + int(0.2 * (max(x) - min(x)))
        y2 = max(y) + int(0.1 * (max(y) - min(y)))
        head_img = img[y1:y2, x1:x2]
        head_img = cv2.resize(head_img, (295, 294))
        gray = cv2.cvtColor(head_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(head_img, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
    #face_cascade = cv2.CascadeClassifier(r'C:\Users\Lenovo\AppData\Local\Programs\Python\Python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
        cv2.imwrite(processed_file_name, cartoon)
        return redirect('/downloadfile/'+ fname +'_processed' + extension)
@app.route("/downloadfile/<filename>", methods = ['GET'])
def download_file(filename):
    return render_template('download.html',value = filename)
@app.route('/return-files/<filename>')
def return_files_tut(filename):
    file_path = 'animated/' + filename
    return send_file(file_path, as_attachment=True)
if __name__ == "__main__":
    app.run(debug=False)
