from flask import Flask, render_template, Response
import cv2
from keras.models import model_from_json
import numpy as np
import os
import random
app = Flask(__name__)

# Load model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load face detector
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels and messages
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}




messages = {
    'angry': [
        "Arre baba, shant ho jao yaar 😡😂",
        "Lagta hai chai nahin piyi aaj ☕🔥",
        "Bhai, bhaukali mat karo! 🐶🤣",
        "Aag mat lagao, warna barfi bhej dunga 🍬😆",
        "Control off? Chill pill lo pehle 💊😤"
    ],
    'disgust': [
        "Yeh kya taste dekha tune? 🤢🤣",
        "Nose bole ‘no no’ 👃😆",
        "Zindagi ka naya tadka: regret 🍋😭",
        "Saaf-safaai ka time hai bhai 🧼👀",
        "Ewww! Face se hi bol raha hai 😝🧟"
    ],
    'fear': [
        "Bhool jao horror, yahan safe zone hai 😱✨",
        "Koi bhoot nahin—sirf Monday hai 👻📅",
        "Auto-correct ne phir evidence mita diya? 😰📱",
        "Darr gaya? Tu bhi na… 😵‍💫",
        "Relax! Monsters aaj chhutti par hain 😅"
    ],
    'happy': [
        "Wah bhai, battery full charge ☀️😁",
        "Smile de de, duniya roshan hogi 🌟😊",
        "Pizza party jeet li kya? 🍕🎉",
        "Monday blues gayab kar di tune 🌈😄",
        "Itni cute mat bano, default lagta hai 💥🧡"
    ],
    'neutral': [
        "Resting face mode on 😐🔥",
        "Yeh expression bhi kamaal hai 🤖🌀",
        "Buffering… brain laga hua hai 🌀🧠",
        "Beparwah. Mast. Neutral 😌",
        "Mood: Chill Maaro ✌️😶"
    ],
    'sad': [
        "Ek cookie toh banta hai na? 🍪🥺",
        "Arey allergy hai boss, rona nahi 🤧🌸",
        "Sad violin baj raha hai 🎻😭",
        "Code crash hua, par tu strong hai 💻😢",
        "Ek virtual hug le le yaar 🤗💙"
    ],
    'surprise': [
        "Kya bolti public? 😲☕",
        "Wi-Fi ne shot maari kya? 📶😮",
        "Anime wala gasp moment! 😳✨",
        "GPA dekh ke shock lag gaya? 😱📊",
        "Drama overload! Plot twist aya! 🫢📖"
    ]
}

# when writing to file, pick randomly:
# f.write(random.choice(messages[prediction_label]))




# Video feed generator
def gen_frames():
    webcam = cv2.VideoCapture(0)
    last_emotion = "waiting"  # Set the initial state to "waiting"

    while True:
        success, im = webcam.read()
        if not success:
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(im, 1.3, 5)

        if len(faces) == 0:  # No face detected
            if last_emotion != "waiting":  # Prevent overwriting if already showing "Waiting"
                last_emotion = "waiting"
                with open("static/message.txt", "w", encoding="utf-8") as f:
                    f.write("Waiting for a face... 😊")
        else:  # Face detected
            for (p, q, r, s) in faces:
                image = gray[q:q + s, p:p + r]
                cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = np.array(image).reshape(1, 48, 48, 1) / 255.0
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]

                if prediction_label != last_emotion:  # If the emotion has changed
                    last_emotion = prediction_label
                    with open("static/message.txt", "w", encoding="utf-8") as f:
                        f.write(random.choice(messages[prediction_label]))

                cv2.putText(im, f'{prediction_label}', (p - 10, q - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', im)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/background')
def background_only():
    return render_template('background_only.html')

if __name__ == '__main__':
    if not os.path.exists("static/message.txt"):
        with open("static/message.txt", "w", encoding="utf-8") as f:
            f.write("Waiting for a face... 😊")
    app.run(debug=True)
