import streamlit as st
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import gdown  # Ensure gdown is in your requirements.txt
import pydeck as pdk
import pandas as pd

# --------------------------
# Custom CSS for Enhanced UI & Background Globe
# --------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
    }
    .stApp {
        background: transparent;
    }
    /* Position the pydeck chart as a fixed background */
    .background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
    }
    /* Foreground content */
    .foreground {
        position: relative;
        z-index: 1;
    }
    h1, h2, h3, h4 {
        text-align: center;
    }
    .stButton button {
        background-color: #ff5733;
        color: white;
        border-radius: 10px;
        padding: 8px 16px;
    }
    .stButton button:hover {
        background-color: #ff2e00;
    }
    div[data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Data for the 3D Drought Globe
# --------------------------
# Example data – adjust with your actual drought dataset.
# Here, each row represents a country with its approximate center coordinates and drought severity (0 to 1).
data = {
    'country': ['USA', 'Brazil', 'India', 'Australia', 'Spain'],
    'lat': [39.8283, -14.2350, 20.5937, -25.2744, 40.4637],
    'lon': [-98.5795, -51.9253, 78.9629, 133.7751, -3.7492],
    'drought_level': [0.2, 0.6, 0.8, 0.5, 0.7]  # 0 = low drought, 1 = severe drought
}
df = pd.DataFrame(data)

# Create a pydeck layer to visualize drought data
drought_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon, lat]',
    get_radius="drought_level * 300000",  # Scale the circle sizes as needed
    get_fill_color="[255, (1 - drought_level) * 255, 0, 200]",
    pickable=True,
)

# Set an initial view state for a global view.
view_state = pdk.ViewState(
    latitude=20,
    longitude=0,
    zoom=1.5,
    pitch=30,
)

# Create the deck.gl chart
r = pdk.Deck(
    layers=[drought_layer],
    initial_view_state=view_state,
    tooltip={"text": "Country: {country}\nDrought Level: {drought_level}"}
)

# Render the globe in a background div
st.markdown('<div class="background">', unsafe_allow_html=True)
st.pydeck_chart(r)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------
# Foreground: Streamlit App Content (Satellite Image Analysis)
# --------------------------
with st.container():
    st.markdown('<div class="foreground">', unsafe_allow_html=True)
    
    st.title("🌍 Satellite Image Rakesh Detection")
    st.markdown("### Upload a satellite image for analysis")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], help="Upload a satellite image")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=350, use_column_width=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sat_model_path = os.path.join(current_dir, "satellite_classifier_finetuned_v2.keras")
            sat_model = tf.keras.models.load_model(sat_model_path)
        except Exception as e:
            st.error(f"Error loading satellite classifier model: {e}")
            st.stop()
        
        def preprocess_image(image_path, target_size):
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            return np.expand_dims(img_array, axis=0)
        
        img_sat = preprocess_image(tmp_path, (64, 64))
        sat_pred = sat_model.predict(img_sat)[0][0]
        
        if sat_pred > 0.5:
            sat_label = "Satellite Image Detected ✅"
        else:
            sat_label = "⚠️ Please upload a valid satellite image!"
        
        st.subheader(sat_label)
        
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
                
                fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
                ax.imshow(image)
                ax.set_title(drought_label, fontsize=14, color='red' if "Drought" in drought_label else 'green', pad=10)
                ax.axis('off')
                plt.tight_layout()
                
                st.success("Analysis Complete!")
                st.pyplot(fig)
                
        os.remove(tmp_path)
    
    st.markdown('</div>', unsafe_allow_html=True)
