import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import gdown
import os
import zipfile

# Google Drive links for `.zip` files
detection_model_zip_url = "https://drive.google.com/uc?id=10-uFlRCIz6_Hxe3bR6RN-D-xt59VRLvz"
grad_cam_model_zip_url = "https://drive.google.com/uc?id=1TM3PQpj6W1iytuC9H53vjJ8Rk9qKocgX"

# Local paths for the `.zip` files and extracted models
detection_model_zip_path = "model_12.zip"
grad_cam_model_zip_path = "model_09.zip"
detection_model_path = "model_12.keras"
grad_cam_model_path = "model_09.h5"

# Function to download files
def download_file(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        gdown.download(url, output_path, quiet=False)

# Function to extract `.zip` files
def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("./")  # Extracts to the current folder
        print(f"{zip_path} extracted successfully.")

# Download and extract detection model
download_file(detection_model_zip_url, detection_model_zip_path)
extract_zip(detection_model_zip_path, detection_model_path)

# Download and extract GRAD-CAM model
download_file(grad_cam_model_zip_url, grad_cam_model_zip_path)
extract_zip(grad_cam_model_zip_path, grad_cam_model_path)

# Load detection model
print("Loading detection model...")
try:
    detection_model = tf.keras.models.load_model(detection_model_path)
    print("Detection model loaded successfully.")
except Exception as e:
    print(f"Error loading detection model: {e}")
    exit()

# Load GRAD-CAM model
print("Loading GRAD-CAM model...")
try:
    grad_cam_model = tf.keras.models.load_model(grad_cam_model_path)
    print("GRAD-CAM model loaded successfully.")
except Exception as e:
    print(f"Error loading GRAD-CAM model: {e}")
    exit()

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function for GRAD-CAM heatmap
def generate_grad_cam(model, image_array, layer_name="conv2d_2"):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )
    except Exception as e:
        print(f"Error accessing layer {layer_name}: {e}")
        return None

    # Compute GRAD-CAM
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    guided_grads = grads[0]
    conv_outputs = conv_outputs[0]

    weights = np.mean(guided_grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam) if np.max(cam) != 0 else cam  # Normalize heatmap
    return heatmap

# Function for combining detection and GRAD-CAM
def detect_and_visualize(image):
    try:
        # Preprocess the image
        image_array = preprocess_image(image)

        # Use detection model for prediction
        prediction = detection_model.predict(image_array)
        result = "Stroke Detected" if prediction[0][0] > 0.5 else "Stroke Not Detected"

        # Generate GRAD-CAM heatmap
        heatmap = generate_grad_cam(grad_cam_model, image_array, layer_name="conv2d_2")
        if heatmap is None:
            return result, "Error generating heatmap. Layer not found."

        # Overlay heatmap on the original image
        original_image = np.array(image.resize((224, 224)))
        plt.figure(figsize=(6, 6))
        plt.imshow(original_image / 255.0)  # Normalize original image
        plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay heatmap
        plt.colorbar(label="Importance")  # Add color bar for scale
        plt.axis('off')

        # Save the heatmap as an image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return result, buf
    except Exception as e:
        print(f"Error during detection or visualization: {e}")
        return "Error in processing", None

# Gradio interface
interface = gr.Interface(
    fn=detect_and_visualize,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Detection Result"), gr.Image(label="Grad-CAM Heatmap")],
    title="Ischemic Stroke Detection with GRAD-CAM",
    description="Upload an MRI image to detect ischemic stroke and visualize regions of interest using GRAD-CAM."
)

# Launch the application
print("Launching interface...")
interface.launch()
