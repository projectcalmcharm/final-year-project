from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import sqlite3
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import librosa
import pickle
import matplotlib.pyplot as plt
#from library.speech_emotion_recognition import *  # Import custom functions for speech emotion recognition
import tempfile
from moviepy.editor import VideoFileClip
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
#from speech_emotion_recognition import SpeechEmotionRecognition
import random

import speech_recognition as sr
import pickle

app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


##model_sub_dir = os.path.join('Models', 'audio.hdf5')

face_model = load_model('Models/model1.h5')   # Load the face emotion recognition model

# Text model for speech emotion recognition
with open('Models/tfidf_vectorizer.pkl', 'rb') as f:
    tfvect = pickle.load(f)

with open('Models/text_model.pkl', 'rb') as f:
    text_model = pickle.load(f)


emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']

database = "database.db"
conn = sqlite3.connect(database)
cursor = conn.cursor()
cursor.execute("create table if not exists custom(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,password TEXT NOT NULL,email TEXT unique,cpassword TEXT,age INTEGER)")
cursor.execute("create table if not exists image(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT, age INTEGER, image blob, image1 blob)")
conn.commit()



@app.route('/')
def index():
    return render_template('index.html')

names_list = []
emails_list = []
ages_list = []


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        try:
            mail = request.form['name']
            password = request.form['password']
            con = sqlite3.connect(database)
            con.row_factory = sqlite3.Row
            cur = con.cursor()

            # Check user credentials
            cur.execute("SELECT * FROM custom WHERE email=? AND password=?", (mail, password))
            data = cur.fetchone()
            print(data[1])

            if data:
                # Store user details in session
                session["name"] = data["name"]
                session["email"] = data["email"]
                session["age"] = data["age"]
                print(session["name"])
                con = sqlite3.connect(database)
                con.row_factory = sqlite3.Row
                cur = con.cursor()
    
                # Fetch the latest image uploaded by the logged-in user
                cur.execute("SELECT image, image1 FROM image WHERE email=? ORDER BY id DESC LIMIT 1", (mail,))
                image_data = cur.fetchone()

                # Convert the image data to base64 for rendering in the template
                image_base64_pie=None
                if image_data:
                    if image_data['image']:
                        image_base64 = base64.b64encode(image_data['image']).decode('utf-8')
                    if image_data['image1']:
                        image_base64_pie = base64.b64encode(image_data['image1']).decode('utf-8')

                # Store the image data in the session
                

                # Append user details to respective lists
                names_list.append(data["name"])
                print(names_list)
                emails_list.append(data["email"])
                ages_list.append(data["age"])
                
                              
                return redirect(url_for("dashboard1", image = image_base64_pie))
            else:
                flash("Username and password mismatch", "danger")
            

        except Exception as e:
            print(f"Error: {str(e)}")
            flash("Check Your Name and Password", "danger")

    return render_template("index.html")

@app.route('/dashboard')
def dashboard():
    user_name = session.get('name')
    user_email = session.get('email')
    user_age = session.get('age')

    # Check if the user is logged in
    if not user_name:
        return redirect(url_for('login'))  # Redirect to login page if user is not logged in

    con = sqlite3.connect(database)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Fetch the latest image uploaded by the logged-in user
    cur.execute("SELECT image, image1 FROM image WHERE name=? ORDER BY id DESC LIMIT 1", (user_name,))
    image_data = cur.fetchone()

    # Convert the image data to base64 for rendering in the template
    image_base64 = None
    image_base64_pie = None
    if image_data and image_data['image']:
        image_base64 = base64.b64encode(image_data['image']).decode('utf-8')
       # Pass the image and user details to the template
    return render_template("dashboard.html", 
                           user={'name': user_name, 'email': user_email, 'age': user_age})

    

@app.route('/dashboard1')
def dashboard1():
    # Get user details from session
    user_name = session.get('name')
    print(user_name)
    user_email = session.get('email')
    user_age = session.get('age')

    # Get the image data from the URL query parameter
    #image_base64 = request.args.get('image')
   # image_base64_pie = session.get('image1')

    # Pass the image and user details to the template
    return render_template("dashboard.html", user={'name': user_name, 'email': user_email, 'age': user_age})


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method=='POST':
        try:
            name=request.form['name']
            password=request.form['password']
            email=request.form['email']
            cpassword=request.form['cpassword']
            age=request.form['age']
            if password == cpassword:                
                con=sqlite3.connect(database)
                cur=con.cursor()
                cur.execute("insert into custom(name,password,email,cpassword,age)values(?,?,?,?,?)",(name,password,email,cpassword,age))
                con.commit()
                flash("Record Added Successfully","success")
            else:
                flash("Password Mismatch","danger")
        except:
            flash("Error in Insert Operation","danger")
        finally:
            return redirect(url_for("index"))
            con.close()

    return render_template('register.html')


import base64

@app.route('/report')
def report():
    # Check if the user is logged in
    if "email" not in session:
        flash("You need to log in first", "warning")
        return redirect(url_for("login"))

    try:
        # Get the logged-in user's email
        user_email = session["email"]
        
        # Connect to the database
        con = sqlite3.connect(database)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        # Fetch records from the 'image' table for the logged-in user's email
        cur.execute("SELECT name, email, age, image, image1 FROM image WHERE email=?", (user_email,))
        data = cur.fetchall()

        if not data:
            flash("No report found for your email.", "warning")
            return redirect(url_for("dashboard1"))

        # Prepare a list of user details and images for the report
        user_images = []
        for row in data:
            image_base64 = base64.b64encode(row["image"]).decode('utf-8') if row["image"] else None
            user_details = {
                "name": row["name"],
                "email": row["email"],
                "age": row["age"],
                "image": image_base64
            }
            user_images.append(user_details)

        # Render the 'report.html' template with the user's images and details
        return render_template("report.html", users=user_images)

    except Exception as e:
        print(f"Error fetching user report: {str(e)}")
        flash("Failed to load your report.", "danger")
        return redirect(url_for("dashboard1"))



def preprocess_face_image(face_image):
    if len(face_image.shape) == 3:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, (224, 224))
    if len(face_image.shape) == 2:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
    face_image = face_image / 255.0
    face_image = np.reshape(face_image, (1, 224, 224, 3))
    return face_image

def analyze_face_emotion(video_path):
    cap = cv2.VideoCapture(video_path)
    emotion_counts = {label: 0 for label in emotion_labels}
    graph_image_blob = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame.shape[-1] == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y + h, x:x + w]
                face_image = preprocess_face_image(face_roi)
                emotion_prediction = face_model.predict(face_image)
                emotion_label = emotion_labels[np.argmax(emotion_prediction)]
                print(emotion_label)
                emotion_counts[emotion_label] += 1
    face_em = list(emotion_counts.values())
   
    cap.release()

    # Plot and save emotion distribution graph
    plt.figure(figsize=(10, 5))
    plt.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
    plt.xlabel("Emotions")
    plt.ylabel("Counts")
    plt.title("Detected Emotions in Video")

    graph_path = os.path.join(app.static_folder, 'emotion_graph.png')
    plt.savefig(graph_path)
    plt.close()

    return graph_path
    
    



@app.route('/videopage', methods=["GET", "POST"])
def video_page():
    return render_template('video.html')


@app.route('/video', methods=["GET", "POST"])
def video():
    if request.method == "POST":
        video_file = request.files.get('video')
        audio_file = request.files.get('audio')

        if not video_file or not audio_file:    
            return jsonify({"error": "No file selected"}), 400


        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)


        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
        audio_file.save(audio_path)

        
        face_value = analyze_face_emotion(video_path)
        print(face_value)
        
        audio_file_path = 'uploads/recorded_audio.wav'
        text_emotion = transcribe_audio_to_text_from_file(audio_file_path)
        print(text_emotion)       
        
        return jsonify({
            "graph_path": face_value,            
            "text_path" : text_emotion
        })              

        
        

    return render_template('video.html')



text_emotion_labels = ["Anxiety", "Bipolar", "Depression", "Normal", "Personality Disorder", "Stress", "Suicidal"]

recommendations = {
    "Anxiety": {
        "Suggestion":"If you're feeling anxious, try taking slow, deep breaths to calm your mind.Engage in a soothing activity like listening to music or taking a walk to help reduce stress",
        "book": "The Anxiety and Phobia Workbook by Edmund J. Bourne",
        "music": "https://www.youtube.com/watch?v=lFcSrYw-ARY"
    },
    "Bipolar": {
        "Suggestion":"Consistent sleep patterns, a balanced routine, and stress management techniques are also beneficial.",
        "book": "The Bipolar Disorder Survival Guide by David J. Miklowitz",
        "music": "https://www.youtube.com/watch?v=JFeVn-hH2mA"
    },
    "Depression": {
        "Suggestion":"Regular exercise, a balanced diet, and staying connected with loved ones can also improve mood and mental well-being.",
        "book": "The Noonday Demon: An Atlas of Depression by Andrew Solomon",
        "music": "https://www.youtube.com/watch?v=ZR7AWSwuWIM",
        "travel": "https://www.google.com/maps/place/Attukad+Waterfalls/@10.0534461,77.0562752,17z/data=!3m1!4b1!4m6!3m5!1s0x3b0799072e70314d:0x15e80e7c92e710da!8m2!3d10.0534408!4d77.0588501!16s%2Fg%2F1q62ccz0d?entry=ttu&g_ep=EgoyMDI0MTIxMS4wIKXMDSoASAFQAw%3D%3D"
    },
    "Normal": {
        "Suggestion":"To maintain well-being, keep a balanced routine with regular exercise and healthy eating. Stay connected with friends and family for emotional support and positive interactions.",
        "book": "Atomic Habits by James Clear",
        "music": "https://www.youtube.com/watch?v=nQA97xS49LQ"
    },
    "Personality Disorder": {
        "Suggestion":"Building a strong support network and focusing on self-awareness can help improve emotional regulation.",
        "book": "The Borderline Personality Disorder Survival Guide by Alexander L. Chapman",
        "music": "https://www.youtube.com/watch?v=QW8bWHtnOOg"
    },
    "Stress": {
        "Suggestion":"To manage stress, practice relaxation techniques like deep breathing or meditation. Regular physical activity and adequate rest can also help reduce stress levels effectively.",
        "book": "The Relaxation and Stress Reduction Workbook by Martha Davis",
        "music": "https://www.youtube.com/watch?v=cospUJJxUqQ",
        "travel":"https://www.google.com/maps/place/Yercaud,+Tamil+Nadu+636601/@11.774813,78.1994054,15z/data=!3m1!4b1!4m6!3m5!1s0x3babf42b4747d5eb:0x1d3ccd9945d5e7ee!8m2!3d11.7747924!4d78.2097052!16zL20vMDMxZHBt?entry=ttu&g_ep=EgoyMDI0MTIxMS4wIKXMDSoASAFQAw%3D%3D"
    },
    "Suicidal": {
        "Suggestion":"Talking to someone you trust, such as a close friend or family member, can also provide comfort and help you feel less alone.",
        "book": "The Suicide Workbook: The Depression and Suicidal Thoughts Workbook by Rebecca H. Williams",
        "music": "https://www.youtube.com/playlist?list=PLZo8MzGmu9_fR2S6sjvFnBCQcm7wFoKCd"  
    }
}

def transcribe_audio_to_text_from_file(audio_file_path):
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)

    # Convert audio to text
    text = recognizer.recognize_google(audio)
    print(f"Transcribed Text: {text}")

    # Predict text emotion using the loaded model
    vectorized_input_data = tfvect.transform([text])
    prediction = text_model.predict(vectorized_input_data)
    print(f"Text Prediction: {prediction}")
    predicted_label = text_emotion_labels[prediction[0]]

    print("Predicted Text Emotion:", predicted_label)
    return predicted_label



@app.route('/output', methods=["GET", "POST"])
def output():
    graph_path = request.args.get('graph_path', None)
    print(graph_path, "graph_path")
    text_emotion = request.args.get('text_path', None)
    print(text_emotion, "text_path")
    
    # Get recommendations based on text_emotion label
    book_recommendation = recommendations.get(text_emotion, {}).get('book', 'No book recommendation available.')
    music_recommendation = recommendations.get(text_emotion, {}).get('music', 'No music recommendation available.')
    Suggestion_recommendation = recommendations.get(text_emotion, {}).get('Suggestion', 'No suggestion_recommendation available.')


    if graph_path and text_emotion:
        return render_template('results.html', 
                               graph_path=graph_path, 
                               text_path=text_emotion,
                               Suggestion_recommendation=Suggestion_recommendation,
                               book_recommendation=book_recommendation,
                               music_recommendation=music_recommendation)
    return "error"


if __name__ == '__main__':
    app.run(debug=False, port=250)
