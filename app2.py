import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

model_05_path = "model_05.keras"
model_06_path = "model_06.keras"

print("Loading models...")
try:
    model_05 = tf.keras.models.load_model(model_05_path)
    model_06 = tf.keras.models.load_model(model_06_path)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

def predict_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    try:
        prediction_05 = model_05.predict(image_array)
        prediction_06 = model_06.predict(image_array)
        
        if prediction_05[0][0] > 0.5 or prediction_06[0][0] > 0.5:
            return "Stroke Detected"
        else:
            return "Stroke Not Detected"
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error in prediction"

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Ischemic Stroke Detection",
    description="Upload an MRI image to detect if ischemic stroke is present."
)

print("Launching interface...")
interface.launch()
