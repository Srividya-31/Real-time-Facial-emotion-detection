import cv2
import numpy as np
from keras.models import model_from_json
from PIL import ImageFont, ImageDraw, Image
import textwrap

# Load model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load face detector
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Label map
labels = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

# Messages based on emotion
messages = {
    'angry': "Take a deep breath, hottie 😡❤️",
    'disgust': "Still cute even with that face 😜💚",
    'fear': "Don't worry cutie, I'm right here 😨✨",
    'happy': "That smile is illegal 😍💫",
    'neutral': "Poker face, but still slaying 🤨🔥",
    'sad': "Smile for me, beautiful 😢💖",
    'surprise': "Ooo what's the tea? 😲☕"
}

# Resize & normalize
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Font path for emoji-compatible font (Windows)
font_path = "C:/Windows/Fonts/seguiemj.ttf"  # If not working, let me know!

# Start webcam
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()
    frame = cv2.resize(frame, (1280, 720))  # Fullscreen size
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        features = extract_features(face)
        pred = model.predict(features)
        label = labels[pred.argmax()]
        message = messages[label]

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Convert frame to PIL
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, 28)

        # Wrap text
        wrapped_text = textwrap.wrap(message, width=30)
        line_height = font.getbbox("hg")[3] - font.getbbox("hg")[1]
        box_width = 600
        box_height = len(wrapped_text) * line_height + 20

        # Draw message box
        draw.rectangle([(20, 20), (20 + box_width, 20 + box_height)], fill=(0, 0, 0, 160))

        # Draw wrapped lines
        y_text = 30
        for line in wrapped_text:
            draw.text((30, y_text), line, font=font, fill=(255, 255, 255))
            y_text += line_height

        # Convert back to OpenCV
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Put label on face
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    cv2.imshow("Mood Detector 💖", frame)
    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
