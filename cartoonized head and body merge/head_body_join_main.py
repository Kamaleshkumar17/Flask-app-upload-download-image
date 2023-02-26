from werkzeug.utils import secure_filename
from flask import Flask,request,redirect,flash,render_template,send_file
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
app = Flask(__name__)
boy_girl_head=r'F:\CV\boy-girl-body'
boy_girl_body=r'F:\CV\boy-girl-head'
processed=r'F:\CV\processed'
app.config['boy_girl_head'] = boy_girl_head
app.config['boy_girl_body'] = boy_girl_body
app.config['processed'] = processed
app.template_folder = 'templates'
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the head and body image parts
        if 'head_image' not in request.files or 'body_image' not in request.files:
            flash('Please upload both head and body images')
            return redirect(request.url)
        head_image = request.files['head_image']
        body_image = request.files['body_image']
        # if user does not select file for either part, browser also
        # submit an empty part without filename
        if head_image.filename == '' or body_image.filename == '':
            flash('Please upload both head and body images')
            return redirect(request.url)
        else:
            os.makedirs(app.config['boy_girl_head'], exist_ok=True)
            os.makedirs(app.config['boy_girl_body'], exist_ok=True)
            head_filename = secure_filename(head_image.filename)
            body_filename = secure_filename(body_image.filename)
            head_image.save(os.path.join(app.config['boy_girl_head'], head_filename))
            body_image.save(os.path.join(app.config['boy_girl_body'], body_filename))
            print("Saved head image to:", os.path.join(app.config['boy_girl_head'], head_filename))
            print("Saved body image to:", os.path.join(app.config['boy_girl_body'], body_filename))
            # redirect to a new route to process the head and body images
            return redirect('/join_head_body/' + head_filename +'/'+ body_filename)
    return render_template('upload_file.html')
@app.route('/join_head_body/<head_image>/<body_image>', methods=['GET'])
def join_head_body(head_image, body_image):
    # Load head and body images
    dirName = 'processed/'
    fname, extension = os.path.splitext(head_image)
    face = cv2.imread(os.path.join(app.config['boy_girl_head'], head_image))
    body = cv2.imread(os.path.join(app.config['boy_girl_body'], body_image))
    result = body.copy()
    body_height, body_width, _ = body.shape
    face_height, face_width, _ = face.shape
    # Resize the head image to match the body dimensions
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
        # Save the processed image
        cv2.imwrite(os.path.join(app.config['processed'], fname + '_processed' + extension), result)
        print("Saved head image to:", os.path.join(app.config['processed'], fname + '_processed' + extension ))
        return redirect('/downloadfile/'+ fname + '_processed' + extension)
@app.route("/downloadfile/<filename>", methods = ['GET'])
def download_file(filename):
    return render_template('download.html',value = filename)
@app.route('/return-files/<filename>')
def return_files_tut(filename):
    file_path = 'processed/' + filename
    return send_file(file_path, as_attachment=True)
