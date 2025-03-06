import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from tensorflow.keras.models import load_model
import tensorflow as tf

# Initialize camera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load the trained model
model = load_model('Model/best_model.h5')

# Parameters
offset = 20
imgSize = 224  # MobileNetV2 input size

# Load class labels
labels = []
with open('Model/labels.txt', 'r') as file:
    labels = [line.strip() for line in file]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Create white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Crop hand image
        imgCrop = img[max(0, y-offset):min(y+h+offset, img.shape[0]), 
                     max(0, x-offset):min(x+w+offset, img.shape[1])]
        
        if imgCrop.size != 0:  # Check if crop is not empty
            aspectRatio = h/w
            
            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
            
            # Prepare image for prediction
            img_array = tf.keras.preprocessing.image.img_to_array(imgWhite)
            img_array = tf.expand_dims(img_array, 0)
            img_array = img_array / 255.0
            
            # Make prediction
            prediction = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Display prediction
            label = labels[predicted_class]
            confidence_pct = confidence * 100
            text = f'{label} ({confidence_pct:.1f}%)'
            
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 1
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Position text well above the hand box
            text_x = x
            text_y = y - 60  # Increased spacing from hand box
            
            # Draw background rectangle for text
            rect_x = text_x - 10
            rect_y = text_y - text_height - 10
            rect_w = text_width + 20
            rect_h = text_height + 20
            cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 0), -1)
            
            # Draw text in red color (BGR format)
            cv2.putText(img, text, 
                       (text_x, text_y), font, 
                       font_scale, (0, 0, 255), thickness)
            
            # Show the processed hand image
            cv2.imshow("Hand", imgWhite)
    
    # Show the main camera feed
    cv2.imshow("Camera", img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
