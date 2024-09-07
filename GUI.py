# Importing necessary liberaries
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image,ImageTk
import numpy as np
from tkinter import Button

# GUI setup
def create_gui():
    window = tk.Tk()
    window.title("Voice-Based Age and Emotion Detection")
    
    def upload_file():
        file_path = filedialog.askopenfilename()
        if file_path:
            process_voice(file_path)
    
    def process_voice(file_path):
        features = extract_features(file_path).reshape(1, -1)
        
        # Gender Detection
        gender = gender_model.predict(features)
        
        if gender == 0:  # If detected gender is female
            messagebox.showinfo("Error", "Upload male voice")
        else:
            # Age Detection
            age = age_model.predict(features)[0]
            
            if age > 60:
                emotion_probs = emotion_model.predict(features)
                emotion = encoder.inverse_transform([np.argmax(emotion_probs)])[0]
                messagebox.showinfo("Result", f"Age: {age}\nEmotion: {emotion}")
            else:
                messagebox.showinfo("Result", f"Age: {age}")
    
    upload_button = tk.Button(window, text="Upload Voice File", command=upload_file)
    upload_button.pack(pady=20)
    
    window.mainloop()

# Start the GUI
create_gui()