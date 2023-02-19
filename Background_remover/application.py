import os
import cv2 
#import imutils
from werkzeug.utils import secure_filename
from flask import Flask,flash,request,redirect,send_file,render_template,send_from_directory, jsonify
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from rembg import remove
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
app = Flask(__name__)
app.template_folder = 'templates'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
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
            return redirect('/process-file/'+ filename)
    return render_template('upload_file.html')
@app.route('/process-file/<filename>', methods = ['GET'])
def process_file(filename):
    dirName = 'processed/'
    try:
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    fname, extension = os.path.splitext(filename)
    processed_file_name = dirName + fname +'_processed' + extension
    file_path = UPLOAD_FOLDER + filename
    img = Image.open(file_path)
    result = remove(img)
    result = result.convert("RGB")
    result.save(processed_file_name, format='JPEG')
    return redirect('/downloadfile/'+ fname +'_processed' + extension)
@app.route("/downloadfile/<filename>", methods = ['GET'])
def download_file(filename):
    return render_template('download.html',value = filename)
@app.route('/return-files/<filename>', methods = ['GET'])
def return_files_tut(filename):
    file_path = 'processed/' + filename
    return send_file(file_path, as_attachment=True)
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)

