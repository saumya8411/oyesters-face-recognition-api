from flask import Flask,jsonify,request,render_template,redirect,flash,url_for
from werkzeug.utils import secure_filename
import mysql.connector
import os
import face_recognition as fr
import datetime
import json
import numpy as np
import time
from flask_swagger_ui import get_swaggerui_blueprint
from dotenv import load_dotenv

load_dotenv()


mydb = mysql.connector.connect(
  host=os.getenv("HOST"),
  user=os.getenv("USER"),
  password=os.getenv("PASSWORD"),
  database=os.getenv("DATABASE")
)

mycursor = mydb.cursor()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Face Recognition"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

# Endpoint for registration
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint for registration
@app.route('/register',methods=["POST"])
def register():
    start_time = time.time()
    global mycursor
    if request.method == 'POST':
        # checking if name already exists
        name = request.form['name']
        mycursor.execute("SELECT NAME FROM users")
        results = mycursor.fetchall()
        for i in results:
           if name in i:
               return jsonify({"status":400,"message":"Name is already taken"})
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return jsonify({"status":400,"message":"File variable not included in request"})
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return jsonify({"status":400,"message":"No image uploaded"})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            ct = datetime.datetime.now()
            try:
                img = fr.load_image_file(path)
                embedding = fr.face_encodings(img)[0]
            except:
                os.remove(path)
                return jsonify({"status":400,"message":"Unable to detect face"})

            x = convertToBinaryData(path)
            mycursor.execute("INSERT INTO users(NAME, TIMESTAMP, IMAGE, EMBEDDING) VALUES (%s,%s,%s,%s)",(name,ct,x,json.dumps(list(embedding))))
            mydb.commit()
            os.remove(path)
            end_time = time.time()
            print(start_time)
            
            print(end_time)

            return jsonify({"status":200,"message":"User details registered","time_taken":end_time-start_time})
    return jsonify({"status":400,"message":"Bad Request"})
    


# Endpoint for verification
@app.route('/verify',methods=["POST"])
def verify():

    start_time = time.time()
    global mycursor
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return jsonify({"status":400,"message":"File variable not included in request"})
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return jsonify({"status":400,"message":"No image uploaded"})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(path)
            ct = datetime.datetime.now()

            try:
                img = fr.load_image_file(path)
                embedding = fr.face_encodings(img)[0]
            except:
                os.remove(path)
                return jsonify({"status":400,"message":"Unable to detect face"})

            x = convertToBinaryData(path)

            mycursor.execute("SELECT NAME,EMBEDDING FROM users")
            results = mycursor.fetchall()
            embeddings = []
            names = []

            for i in results:
                names.append(i[0])
                embeddings.append(list(json.loads(i[1])))
            results = fr.face_distance(embeddings, embedding)
            print(names)
            detectedFace = names[np.argmin(results)]
            ct = datetime.datetime.now()

            mycursor.execute("INSERT INTO logs(NAME, TIMESTAMP) VALUES (%s,%s)",(detectedFace,ct))

            mydb.commit()
            os.remove(path)
            end_time = time.time()
            
            return jsonify({"status":200,"message":"User verified","name":detectedFace,"time_taken":end_time-start_time})
    return jsonify({"status":400,"message":"Bad Request"})



if __name__ == '__main__':
    app.secret_key = os.getenv("SECRET_KEY")
    app.run(debug = False,host=os.getenv("APP_HOST"),port=os.getenv("PORT"))