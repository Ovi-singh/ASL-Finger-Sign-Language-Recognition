import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('asl_model.h5')

class_names = ['A', 'B', 'S', 'D', 'E', 'M', 'G']

# Create the main window
root = tk.Tk()
root.title("Sign Language Letter Recognition")

# Load the background image and set the window size to match
bg_image_path = "C:/Users/dell/Downloads/one last try/gui.png"
bg_image = Image.open(bg_image_path)
bg_image = bg_image.resize((1024, 512))  # Resize to match original dimensions
bg_image_tk = ImageTk.PhotoImage(bg_image)

# Set window size and background
root.geometry("1024x512")

# Display background image on a canvas
canvas = tk.Canvas(root, width=1024, height=512)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, anchor="nw", image=bg_image_tk)

# Function for uploading and predicting an image
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return
    
    try:
        # Load and process the image
        img = Image.open(file_path)
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Predict using the trained model
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]  # Get the class with the highest probability
        
        # Display the result
        prediction_label.config(text=f"Prediction: {predicted_class}")
        
        # Display the image
        img = Image.open(file_path)
        img = img.resize((100, 100))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Place upload button and prediction label without overlapping top/bottom text
upload_button = tk.Button(root, text="Upload Image", command=upload_image, font=('Helvetica', 14), bg="#415A77", fg="white", padx=10, pady=5)
canvas.create_window(512, 220, window=upload_button)  # Adjust y-position if needed

# Label for prediction result
prediction_label = tk.Label(root, text="Prediction: ", font=('Helvetica', 16), bg="white", fg="black")
canvas.create_window(512, 270, window=prediction_label)  # Adjust y-position if needed

# Label for showing the uploaded image
image_label = tk.Label(root, bg="white")
canvas.create_window(512, 340, window=image_label)  # Adjust y-position if needed

# Start the GUI loop
root.mainloop()
