import os
import cv2
from werkzeug.utils import secure_filename
from flask import Flask,flash,request,redirect,send_file,render_template,send_from_directory,url_for,jsonify
from PIL import Image, ImageDraw, ImageFont
import datetime
import numpy as np
import zipfile
import shutil
import time
import threading
import face_recognition_models
UPLOAD_FOLDER = r'/home/kamalesh042018/demoapplication/uploads/'
PROCESSED_FOLDER = r'/home/kamalesh042018/demoapplication/processed/'
UPLOAD_VIDEO_AUDIO_FOLDER = r'/home/kamalesh042018/demoapplication/upload-video-audio/'
NO_FACE_DETECTED = r'/home/kamalesh042018/demoapplication/no-face-detected/'
ZIPFILE_FOLDER = r'/home/kamalesh042018/demoapplication/zipfilefol/'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'mp4', 'mp3', 'png'])
app = Flask(__name__)
app.template_folder = 'templates'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['UPLOAD_VIDEO_AUDIO_FOLDER'] = UPLOAD_VIDEO_AUDIO_FOLDER
app.config['NO_FACE_DETECTED'] = NO_FACE_DETECTED
app.config['ZIPFILE_FOLDER'] = ZIPFILE_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            return jsonify({'msg': 'No file part in the request'}), 400
        files = request.files.getlist("files[]")
        filenames=[]
        filenames_nfd=[]
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                if filename.split('.')[1] == 'mp4' or filename.split('.')[1] == 'mp3':
                    os.makedirs(app.config['UPLOAD_VIDEO_AUDIO_FOLDER'], exist_ok=True)
                    file.save(os.path.join(app.config['UPLOAD_VIDEO_AUDIO_FOLDER'], filename))
                    filenames.append(filename)
                    return redirect('/download-video-audio/'+ ','.join(filenames))
                else:
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    print("saved file successfully")
                    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    face_cascde = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    faces = face_cascde.detectMultiScale(gray, 1.3, 5)
                    if len(faces) > 0:
                        filenames.append(filename)
                    else:
                        filenames_nfd.append(filename)
                        os.makedirs(app.config['NO_FACE_DETECTED'], exist_ok=True)
                        shutil.move(os.path.join(app.config['UPLOAD_FOLDER'], filename), os.path.join(app.config['NO_FACE_DETECTED'], filename))
        if len(filenames) > 0:
            return redirect('/process-file/'+','.join(filenames))
        elif len(filenames_nfd) > 0:
            return redirect('/download-nfd/'+ ','.join(filenames_nfd))
    return render_template('upload_file.html')                
@app.route('/download-video-audio/<filenames>', methods=['GET'])
def download_video_audio(filenames):
    filenames = filenames.split(',')
    for filename in filenames:
        return redirect('/downloadfile/'+ filename)
@app.route('/download-nfd/<filenames>', methods=['GET'])
def download_nfd(filenames):
    filenames = filenames.split(',')
    for filename in filenames:
        fname, extension = os.path.splitext(filename)
        return redirect('/downloadfile/'+ fname +'_processed' + extension)
@app.route('/process-file/<filenames>', methods = ['GET'])
def process_file(filenames):
    dirName = r'/home/kamalesh042018/demoapplication/processed/'
    try:
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    for filename in filenames.split(','):
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        mask = np.zeros(img.shape[:2],np.uint8)
        bgModel = np.zeros((1,65),np.float64)
        fgModel = np.zeros((1,65),np.float64)
        rect = (20,20,img.shape[1]-20,img.shape[0]-20)
        cv2.grabCut(img,mask,rect,bgModel,fgModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)
        print("background removed successfully")
        fname, extension = os.path.splitext(filename)
        processed_file_name = dirName + fname +'_processed' + extension
        date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((0, 0),date ,(255,255,255),font=font)
        draw.text((0, image.size[1]-10),date ,(255,255,255),font=font)
        image.save(processed_file_name)
    return redirect('/downloadfile/'+ fname +'_processed' + extension)
@app.route("/downloadfile/<filename>", methods = ['GET'])
def download_file(filename):
    return render_template('download.html',value = filename)
@app.route('/return-files/<filename>', methods = ['GET'])
def return_files_tut(filename):
    if filename.split('.')[1] == 'mp4' or filename.split('.')[1] == 'mp3':
        path = r'/home/kamalesh042018/demoapplication/upload-video-audio/' + filename
        return send_file(path, as_attachment=True)
    else:
        os.makedirs(app.config['ZIPFILE_FOLDER'], exist_ok=True)
        #zip_file = zipfile.ZipFile(r'/home/kamalesh042018/demoapplication/zipfilefol/'+ fname +'_processed' + extension + '.zip', 'w')
        zip_file = zipfile.ZipFile(r'/home/kamalesh042018/demoapplication/zipfilefol/'+ filename+ '.zip', 'w')
    PROCESSED_FOLDER = r'/home/kamalesh042018/demoapplication/processed/'
    if os.path.exists(PROCESSED_FOLDER):
        for file in os.listdir(PROCESSED_FOLDER):
            if file.endswith('.jpg') or file.endswith('.png'):
                zip_file.write(os.path.join(PROCESSED_FOLDER, file))
    NO_FACE_DETECTED = r'/home/kamalesh042018/demoapplication/no-face-detected/'
    if os.path.exists(NO_FACE_DETECTED):
        for file in os.listdir(NO_FACE_DETECTED):
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                zip_file.write(os.path.join(NO_FACE_DETECTED, file))
    zip_file.close()
    UPLOAD_FOLDER = r'/home/kamalesh042018/demoapplication/uploads/'
    if os.path.exists(UPLOAD_FOLDER):
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    if os.path.exists(PROCESSED_FOLDER):
        for file in os.listdir(PROCESSED_FOLDER):
            file_path = os.path.join(PROCESSED_FOLDER, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    if os.path.exists(NO_FACE_DETECTED):
        for file in os.listdir(NO_FACE_DETECTED):
            file_path = os.path.join(NO_FACE_DETECTED, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    UPLOAD_VIDEO_AUDIO_FOLDER = r'/home/kamalesh042018/demoapplication/upload-video-audio/'
    if os.path.exists(UPLOAD_VIDEO_AUDIO_FOLDER):
        for file in os.listdir(UPLOAD_VIDEO_AUDIO_FOLDER):
            file_path = os.path.join(UPLOAD_VIDEO_AUDIO_FOLDER, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    path = r'/home/kamalesh042018/demoapplication/zipfilefol/' + filename +'.zip'
    return send_file(path, as_attachment=True)
ZIPFILE_FOLDER = r'/home/kamalesh042018/demoapplication/zipfilefol/'
def check_for_zip_files():
    files = os.listdir(ZIPFILE_FOLDER)
    for file in files:
        if file.endswith('.zip'):
            os.remove(os.path.join(ZIPFILE_FOLDER, file))
timer = threading.Timer(80.0, check_for_zip_files)
timer.start()
if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port= 8080)