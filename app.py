import os
import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

detection_model_path = "model_12.keras"
grad_cam_model_path = "model_09.h5"

if not os.path.exists(detection_model_path) or not os.path.exists(grad_cam_model_path):
    exit()

detection_model = tf.keras.models.load_model(detection_model_path)
grad_cam_model = tf.keras.models.load_model(grad_cam_model_path)

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def generate_grad_cam(model, image_array, layer_name="conv2d_2"):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
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
    heatmap = cam / np.max(cam) if np.max(cam) != 0 else cam
    return heatmap

def predict_and_visualize(image):
    try:
        image_array = preprocess_image(image)
        prediction = detection_model.predict(image_array)
        result = "Infected" if prediction[0][0] > 0.5 else "Not Infected"
        heatmap = generate_grad_cam(grad_cam_model, image_array, layer_name="conv2d_2")
        original_image = np.array(image.resize((224, 224)))
        plt.figure(figsize=(6, 6))
        plt.imshow(original_image / 255.0)
        plt.imshow(heatmap, cmap="jet", alpha=0.5)
        plt.colorbar(label="Importance")
        plt.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return result, buf
    except:
        return "Error in processing", None

interface = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Detection Result"), gr.Image(label="Grad-CAM Heatmap")],
    title="Infectious Disease Detection with Grad-CAM",
    description="Upload a medical image to detect if the subject is infected and visualize regions of interest using Grad-CAM."
)

interface.launch()
