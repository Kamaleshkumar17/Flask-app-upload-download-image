from flask import Flask, render_template, request, redirect, url_for,send_file
from werkzeug.utils import secure_filename
import cv2
#from sklearn.cluster import KMeans
from imutils import face_utils
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from rembg import remove
import numpy as np
import os
#import matplotlib.pyplot as plt
import dlib
import numpy as np
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
ANIMATED_FOLDER='animated/'
OUTPUTS_FOLDER='outputs/'
BG_REM='bgrem/'
app.template_folder = 'templates'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANIMATED_FOLDER'] = ANIMATED_FOLDER
app.config['outputs'] = OUTPUTS_FOLDER
app.config['bg_rem'] = BG_REM
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
    fil_path = BG_REM + filename
    dirName = 'bgrem/'
    diName = 'animated/'
    try:
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    fname, extension = os.path.splitext(filename)
    processed_file_name = dirName + filename
    proceed_file_name = diName + fname +"_processed"+extension
    image = Image.open(file_path)
    res = remove(image)
    res = res.convert("RGB")
    res.save(processed_file_name, format='JPEG')
    img = cv2.imread(fil_path)
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
        cv2.imwrite(proceed_file_name, cartoon)
        return redirect('/join_head/'+ fname +'_processed' + extension)
        #return redirect('/downloadfile/'+ fname +'_processed' + extension)
@app.route("/join_head/<filename>", methods = ['GET'])
def join_the_head(filename):
    fname, extension = os.path.splitext(filename)
    head_image = cv2.imread(os.path.join(app.config['ANIMATED_FOLDER'], filename))
    face_cascade = cv2.CascadeClassifier(r'C:\Users\Lenovo\AppData\Local\Programs\Python\Python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(head_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    x, y, w, h = faces[0]
    face = head_image[y:y+h, x:x+w]
    face = cv2.imread(os.path.join(app.config['ANIMATED_FOLDER'], filename))
    body_image = 'body/4.png'
    body = cv2.imread(body_image)
    result = body.copy()
    body_height, body_width, _ = body.shape
    face_height, face_width, _ = face.shape
    if face_width > body_width:
        face = cv2.resize(face, (body_width - 40, (body_width - 40) * face_height // face_width))
        face_height, face_width, _ = face.shape
    elif face_height > body_height:
        face = cv2.resize(face, ((body_height - 230) * face_width // face_height, body_height - 230))
        face_height, face_width, _ = face.shape
    offset_x = 20
    offset_y = -280
    start_x = (body_width - face.shape[1]) // 2 + offset_x
    start_y = (body_height - face.shape[0]) // 2 + offset_y
    if start_x >= 0 and start_y >= 0 and start_x + face_width <= body_width and start_y + face_height <= body_height:
        roi = result[start_y: start_y+face_height, start_x: start_x+face_width]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(face_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        result_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        face_fg = cv2.bitwise_and(face, face, mask=mask)
        result[start_y: start_y+face_height, start_x: start_x+face_width] = cv2.add(result_bg, face_fg)
        cv2.imwrite(os.path.join(app.config['outputs'], fname + '_processed' + extension), result)
        return redirect('/downloadfile/'+ fname +'_processed' + extension)
@app.route("/downloadfile/<filename>", methods = ['GET'])
def download_file(filename):
    return render_template('download.html',value = filename)
@app.route('/return-files/<filename>')
def return_files_tut(filename):
    file_path = 'outputs/' + filename
    return send_file(file_path, as_attachment=True)
if __name__ == "__main__":
    app.run(debug=False)
