import streamlit as st
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import gdown  # Ensure gdown is in your requirements.txt

# --------------------------
# Helper Functions
# --------------------------
def load_and_preprocess_for_satellite(image_path, target_size=(64, 64)):
    """Load image and preprocess for the satellite classifier."""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_and_preprocess_for_drought(image_path, target_size=(65, 65)):
    """Load image and preprocess for the drought detection model."""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def download_drought_model(model_path, drive_url):
    """Download the drought detection model from Google Drive if not present."""
    if not os.path.exists(model_path):
        st.info("Downloading drought detection model from Google Drive...")
        gdown.download(drive_url, model_path, quiet=False)

# --------------------------
# Streamlit App
# --------------------------
st.title("🌍 Satellite Image Drought Detection")
st.markdown("Upload a satellite image for analysis")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image with PIL
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # --------------------------
    # 1. Satellite Classification
    # --------------------------
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
        sat_model_path = os.path.join(current_dir, "satellite_classifier_finetuned_v2.keras")
        sat_model = tf.keras.models.load_model(sat_model_path)
    except Exception as e:
        st.error(f"Error loading satellite classifier model: {e}")
        st.stop()

    img_sat = load_and_preprocess_for_satellite(tmp_path, target_size=(64, 64))
    sat_pred = sat_model.predict(img_sat)[0][0]
    
    # Classification based on prediction
    if sat_pred > 0.5:
        sat_label = "Satellite"
    else:
        sat_label = "Non Satellite"
    
    if sat_label != "Satellite":
        st.error("⚠️ Please upload a valid satellite image!")
    else:
        # --------------------------
        # 2. Drought Detection (only if image is a satellite image)
        # --------------------------
        if st.button("Analyze for Drought"):
            with st.spinner("Processing..."):
                try:
                    # Define local path and Google Drive URL for the drought model
                    drought_model_path = os.path.join(current_dir, "drought_detection_finetuned.keras")
                    drive_url = "https://drive.google.com/uc?id=1PKrPIwDs97hH_iAsz8CYom3GWb2FC3Od"
                    download_drought_model(drought_model_path, drive_url)
                    drought_model = tf.keras.models.load_model(drought_model_path)
                except Exception as e:
                    st.error(f"Error loading drought detection model: {e}")
                    st.stop()
                
                img_drought = load_and_preprocess_for_drought(tmp_path, target_size=(65, 65))
                drought_pred = drought_model.predict(img_drought)[0][0]
                
                # Drought prediction classification
                if drought_pred < 0.5:
                    drought_label = "Drought Detected"
                else:
                    drought_label = "No Drought Detected"
                
                st.success("Analysis Complete!")
                
                # Display result with matplotlib
                fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
                ax.imshow(image)
                ax.set_title(drought_label, fontsize=12, color='red' if "Drought" in drought_label else 'green', pad=6)
                ax.axis('off')
                
                plt.tight_layout(pad=0.5)
                fig.savefig("temp_result.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                result_image = Image.open("temp_result.png")
                st.image(result_image, width=350)
    
    # Remove the temporary file
    os.remove(tmp_path)
