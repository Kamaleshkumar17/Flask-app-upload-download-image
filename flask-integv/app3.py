import os
import cv2
from werkzeug.utils import secure_filename
from flask import Flask,flash,request,redirect,send_file,render_template,send_from_directory,url_for
from PIL import Image, ImageDraw, ImageFont
import datetime
import time
import numpy as np
UPLOAD_FOLDER = r'F:/flask-integv/uploads/'
app = Flask(__name__)
app.template_folder = 'templates'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Upload API
@app.route('/', methods=['GET','POST'])
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
            #return redirect('/process-file/' + filename)
            return redirect(url_for('process_file', filename=filename))
    return render_template('upload_file.html')
@app.route('/process-file/<filename>', methods = ['GET'])
def process_file(filename):
    #file_path = UPLOAD_FOLDER + filename
    # process file
    # save processed file
    # Create a new directory if not exists
    dirName = r'F:/flask-integv/processed/'
    try:
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    fname, extension = os.path.splitext(filename)
    # processed_file_name
    processed_file_name = dirName + fname +'_processed' + extension
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    mask = np.zeros(img.shape[:2],np.uint8)
    bgModel = np.zeros((1,65),np.float64)
    fgModel = np.zeros((1,65),np.float64)
    rect = (20,20,img.shape[1]-20,img.shape[0]-20)
    cv2.grabCut(img,mask,rect,bgModel,fgModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    print("background removed successfully")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(r'C:/Users/Lenovo/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)
    #cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), img)
    # Get the date and save it in the image
    date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # Process the file
    #img = Image.open(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #image=Image.open(processed_file_path)
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    #font = ImageFont.truetype(ft, 10)
    draw.text((0, 0),date ,(255,255,255),font=font)
    draw.text((0, image.size[1]-10),date ,(255,255,255),font=font)
    image.save(processed_file_name) 
    # redirect to download
    return redirect('/downloadfile/'+ fname +'_processed' + extension)
# Download API
@app.route('/downloadfile/<filename>', methods = ['GET'])
def download_file(filename):
    return render_template('download.html',value = filename)
@app.route('/return-files/<filename>', methods = ['GET'])
def return_files_tut(filename):
    file_path = r'F:/flask-integv/processed/' + filename
    return send_file(file_path, as_attachment=True)    

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port= 5000)

