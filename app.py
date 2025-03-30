import streamlit as st
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import gdown  # Ensure gdown is in your requirements.txt
import base64

# --------------------------
# Function to Encode Image in Base64
# --------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Encode the drought heatmap image
background_image_path = "Screenshot 2025-03-30 200633.png"  # Ensure this file is in the same directory as your script.
image_base64 = get_base64_image(background_image_path)

# --------------------------
# Custom CSS for Enhanced UI with Responsive Background Image
# --------------------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(
            rgba(255,255,255,0.5), 
            rgba(255,255,255,0.5)
        ), url("data:image/png;base64,{image_base64}");
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #333; /* Dark text for readability */
    }}

    /* Responsive adjustments for smaller screens */
    @media (max-width: 768px) {{
        .stApp {{
            background-attachment: scroll;
        }}
    }}

    /* Center headings */
    h1, h2, h3, h4 {{
        text-align: center;
    }}

    /* Style the buttons */
    .stButton button {{
        background-color: #ff5733;
        color: white;
        border-radius: 10px;
        padding: 8px 16px;
        border: none;
    }}
    .stButton button:hover {{
        background-color: #ff2e00;
    }}

    /* Style the file uploader box */
    div[data-testid="stFileUploader"] {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Streamlit App Header
# --------------------------
st.title("🌍 Satellite Image Drought Detection")
st.markdown("### Upload a satellite image for analysis")

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose an image", 
    type=["jpg", "jpeg", "png"], 
    help="Upload a satellite image"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=350, use_column_width=True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # --------------------------
    # Load Satellite Classifier Model
    # --------------------------
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sat_model_path = os.path.join(current_dir, "satellite_classifier_finetuned_v2.keras")
        sat_model = tf.keras.models.load_model(sat_model_path)
    except Exception as e:
        st.error(f"Error loading satellite classifier model: {e}")
        st.stop()
    
    # --------------------------
    # Preprocessing Function
    # --------------------------
    def preprocess_image(image_path, target_size):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    
    # --------------------------
    # Classify Satellite Image
    # --------------------------
    img_sat = preprocess_image(tmp_path, (64, 64))
    sat_pred = sat_model.predict(img_sat)[0][0]
    
    if sat_pred > 0.5:
        sat_label = "Satellite Image Detected ✅"
    else:
        sat_label = "⚠️ Please upload a valid satellite image!"
    
    st.subheader(sat_label)
    
    # --------------------------
    # If Satellite Image, Analyze for Drought
    # --------------------------
    if sat_pred > 0.5 and st.button("Analyze for Drought", key="drought_analysis"):
        with st.spinner("Analyzing Image..."):
            try:
                drought_model_path = os.path.join(current_dir, "drought_detection_finetuned.keras")
                drive_url = "https://drive.google.com/uc?id=1PKrPIwDs97hH_iAsz8CYom3GWb2FC3Od"
                if not os.path.exists(drought_model_path):
                    gdown.download(drive_url, drought_model_path, quiet=False)
                drought_model = tf.keras.models.load_model(drought_model_path)
            except Exception as e:
                st.error(f"Error loading drought detection model: {e}")
                st.stop()
            
            img_drought = preprocess_image(tmp_path, (65, 65))
            drought_pred = drought_model.predict(img_drought)[0][0]
            
            drought_label = "🌿 No Drought Detected" if drought_pred >= 0.5 else "🔥 Drought Detected!"
            
            # Display the results on the image
            fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
            ax.imshow(image)
            ax.set_title(
                drought_label, 
                fontsize=14, 
                color='red' if "Drought" in drought_label else 'green', 
                pad=10
            )
            ax.axis('off')
            plt.tight_layout()
            
            st.success("Analysis Complete!")
            st.pyplot(fig)
            
    os.remove(tmp_path)
