import os
from werkzeug.utils import secure_filename
import numpy as np
import dlib
import tensorflow as tf
# from flask_mysqldb import MySQL
from PIL import Image, ImageOps, ImageEnhance
from flask import Flask, jsonify, request
from auth import auth
from joblib import dump, load

app = Flask('predict face')
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_FILE'] = 'models/'
app.config['MYSQL_HOST'] = '34.128.70.7'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'capstone'
app.config['MYSQL_DB'] = 'capstone-db'
# app.secret_key = 'your secret key'

# mysql = MySQL(app)
predictor = tf.keras.models.load_model(app.config['MODEL_FILE'], compile=False, safe_mode=False)
detector = dlib.get_frontal_face_detector()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# def image_to_base64(image):
#     # Convert PIL image to base64 for displaying in HTML
#     buffered = BytesIO()
#     image.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#     return f"data:image/jpeg;base64,{img_str}"

def convert_and_trim_bb(image, rect):
  startX = 0
  startY = 0
  w = 0
  h = 0
  if len(rect) > 0:
    rect = rect[0]
    # extract the starting and ending (x, y)-coordinates of the bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
	  # return our bounding box coordinates

  return (startX, startY, w, h)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(clip_limit)
    return image

def face_detector(image):
  image = Image.open(image)
  image_array = np.array(image)
  faces = detector(image_array)
  crop = convert_and_trim_bb(image_array, faces)

  if sum(crop) > 1 :
    
    cropped_image = image.crop((crop[0],crop[1],crop[0]+crop[2],crop[1]+crop[3]))
    resize_image = cropped_image.resize((160,160))
    photo = resize_image.convert('L')
    enhancer = ImageEnhance.Sharpness(photo)
    photo = enhancer.enhance(2)
    photo = ImageOps.equalize(photo)
    photo = apply_clahe(photo)
    img = np.around((np.array(resize_image) / 255.), decimals=12)
    data_face = np.expand_dims(img, axis=0)

    return data_face
  else:
     return None

def predict_face(image):
  vector = predictor.predict(image)
  return vector

@app.route("/")
def index():
    return jsonify({
       "status": {
          "code": 200,
          "message": "Success fetching the API"
       },
       "data": None
    }), 200


@app.route("/verification", methods=["POST"])
# @auth.login_required()
def preprocessing():
    if request.method == "POST":
      image = request.files["img"]

      if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        face = face_detector(image)
        
        if type(face) == type(None):
           return jsonify({
          "status": {
              "code": 200,
              "message": "request success, face undetected"
          },
          "data": None
        }),200
        else :
          vector_face = predict_face(face)
          # cur = mysql.connection.cursor()
          # cur.execute("SELECT * FROM UserDataModel")
          # data = cur.fetchall()
          # if_clf = load(data[0][1])
          # result = if_clf.predict(vector_face)
          return jsonify({
            "status": {
                "code": 200,
                "message": "request success, face detected"
            },
            "data": f"result,{vector_face}"
          }),200
      
      else:
         return jsonify({
          "status": {
              "code": 422,
              "message": "request success, wrong file extension"
          },
          "data": None
        })
    
    else:
      return jsonify({
        "status": {
            "code": 405,
            "message": "method not allowed"
        },
        "data": None
      })
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

