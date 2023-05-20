import os
import cv2
import imutils
from werkzeug.utils import secure_filename
from flask import Flask,flash,request,redirect,send_file,render_template,send_from_directory, jsonify
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
#from colorthief import ColorThief
from tkinter import Tk, PhotoImage, Button, Label
import datetime
import numpy as np
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
FINALOUT = 'finalout/'
object_cascade = cv2.CascadeClassifier(r'C:\Users\Lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier(r'C:\Users\Lenovo\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
app = Flask(__name__)
app.template_folder = 'templates'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['FINALOUT'] = FINALOUT
# Upload API
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('no file')
            return redirect(request.url)
        file = request.files['file']
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
    return render_template('upload_file.html')
@app.route('/process-file/<filename>', methods = ['GET'])
def process_file(filename):
  file_path = UPLOAD_FOLDER + filename
  dirName = 'processed/'
  try:
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
  except FileExistsError:
    print("Directory " , dirName ,  " already exists")
  fname, extension = os.path.splitext(filename)
  processed_file_name = dirName + fname +'_processed' + extension
  net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
  classes = []
  with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
  layer_names = net.getLayerNames()
  output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
  img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
  colors = np.random.uniform(0, 255, size=(len(classes), 3))
  img = cv2.resize(img, None, fx=0.4, fy=0.4)
  height, width, channels = img.shape
  blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
  net.setInput(blob)
  outs = net.forward(output_layers)
  class_ids = []
  confidences = []
  boxes = []
  for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
  indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
  font = cv2.FONT_HERSHEY_PLAIN
  img_with_boxes = img.copy()
  for i in range(len(boxes)):
    if i in indexes:
      x, y, w, h = boxes[i]
      cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 255, 255), 2)
  #cv2.imwrite(f'{UPLOAD_FOLDER}/{filename}_with_boxes.png', img_with_boxes)
  root = Tk()
  #img_tk = PhotoImage(file=f'{UPLOAD_FOLDER}/{filename}_with_boxes.png')
  #img_tk = PhotoImage(img_with_boxes)
  retval, buffer = cv2.imencode('.png', img_with_boxes)
  img_tk = PhotoImage(data=buffer.tobytes())
  label = Label(root, image=img_tk)
  label.pack()
  selected_boxes = []
  def on_click(event):
    x, y = event.x, event.y
    for i in range(len(boxes)):
      if i in indexes:
        box_x, box_y, box_w, box_h = boxes[i]
        if (x > box_x) and (x < box_x + box_w) and (y > box_y) and (y < box_y + box_h):
          if i not in selected_boxes:
            selected_boxes.append(i)
            cv2.rectangle(img_with_boxes, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 255), 2)
          else:
              selected_boxes.remove(i)
              cv2.rectangle(img_with_boxes, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)
              img_tk = PhotoImage(file=f'{UPLOAD_FOLDER}/{filename}_with_boxes.png')
              label.configure(image=img_tk)
              label.image = img_tk
  def on_press_enter(event):
    fname, extension = os.path.splitext(filename)
    cv2.imwrite(f'{PROCESSED_FOLDER}/{fname}_with_boxes_red.png', img_with_boxes)
    original_img = cv2.imread(os.path.join(app.config['PROCESSED_FOLDER'], fname+'_with_boxes_red.png'))
    output_img = original_img.copy()
    pixels = original_img.reshape((-1,3))
    mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    for i in selected_boxes:
      box_x, box_y, box_w, box_h = boxes[i]
      mask[box_y:box_y+box_h, box_x:box_x+box_w] = 255
    surrounding_colors = pixels[np.where(np.all(pixels != [0, 0, 255], axis=1))]
    colors,counts = np.unique(surrounding_colors, axis=0, return_counts=True)
    most_common_color = colors[np.argmax(counts)]
    most_common_color = tuple(map(int,most_common_color))
    output_img[np.where(mask==255)] = most_common_color
    for i in selected_boxes:
      box_x, box_y, box_w, box_h = boxes[i]
      cv2.rectangle(output_img, (box_x, box_y), (box_x + box_w, box_y + box_h), most_common_color, 2)
    output_img = cv2.inpaint(output_img, mask, 23, cv2.INPAINT_NS)
    #output_img = cv2.inpaint(output_img, mask, cv2.INPAINT_TELEA, 3)
    #cv2.imwrite(f'{UPLOAD_FOLDER}/{filename}_with_background.png', output_img)
    #output_img = cv2.inpaint(output_img, mask, 2, cv2.INPAINT_TELEA)
    cv2.imwrite(f'{FINALOUT}/{fname + extension}', output_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    root.destroy()
  label.bind("<Button-1>", on_click)
  root.bind("<Return>", on_press_enter)
  root.mainloop()
  return redirect('/downloadfile/'+ fname + extension)
@app.route("/downloadfile/<filename>", methods = ['GET'])
def download_file(filename):
    return render_template('download.html',value = filename)
@app.route('/return-files/<filename>', methods = ['GET'])
def return_files_tut(filename):
  file_path = 'finalout/' + filename
  return send_file(file_path, as_attachment=True)
if __name__ == "__main__":
  app.run(debug=False)
uploads_folder = r'F:\Imageprocessor\uploads'
for file_name in os.listdir(uploads_folder):
  file_path = os.path.join(uploads_folder, file_name)
  if os.path.isfile(file_path):
    os.unlink(file_path)
processed_folder = r'F:\Imageprocessor\processed'
for file_name in os.listdir(processed_folder):
  file_path = os.path.join(processed_folder, file_name)
  if os.path.isfile(file_path):
    os.unlink(file_path)
finalout_folder = r'F:\Imageprocessor\finalout'
for file_name in os.listdir(finalout_folder):
  file_path = os.path.join(finalout_folder, file_name)
  if os.path.isfile(file_path):
    os.unlink(file_path)



                      

                             