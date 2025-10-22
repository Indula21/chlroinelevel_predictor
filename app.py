# app.py
import numpy as np
import streamlit as st
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="LOOCV & RGB Analyzer", layout="wide")
st.title("Machine Learning & Image RGB Analyzer")

# Tabs for two parts of your code
tab1, tab2 = st.tabs(["🎨 RGB Image Analyzer", "🔎 Algoritham"])


# ---------------- Tab 2: LOOCV ----------------
with tab2:
    st.header("📊Calibration Samples ")

    # Calibration data
    X = np.array([
        [254,254,6],
        [254,254,52],
        [254,254,3],
        [254,254,64],
        [254,254,72],
        [254,254,169]
    ])
    y = np.array([599.814, 449.8605,649.7985,399.876,299.907,249.9225])

    model = RandomForestRegressor(n_estimators=400, random_state=42)

    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)

    st.subheader("Actual Vs Predicted Concentration")
    for actual, pred in zip(y, y_pred):
        st.write(f"Actual: {actual:.6f}, Predicted: {pred:.6f}")

    # Metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    st.subheader("Model performance")
    st.write(f"RMSE: {rmse:.6f}")
    st.write(f"R²:   {r2:.6f}")

    # Predict new sample
    st.subheader("Predict New Sample")
    r = st.number_input("R value")
    g = st.number_input("G value")
    b = st.number_input("B value")

    new_rgb = np.array([[r, g, b]])
    model.fit(X, y)
    predicted_conc = model.predict(new_rgb)
    st.markdown(
    f"""
    <div style="
        background-color:#1b1b1b;
        border-left:6px solid #00e676;
        padding:12px 18px;
        border-radius:8px;
        color:#e0e0e0;
        font-family:monospace;
    ">
        <b>🧪 Predicted Chlorine Concentration</b><br>
        <span style='color:#81c784'>RGB:</span> {new_rgb.tolist()}<br>
        <span style='color:#00e676'>Predicted Concentration:</span>
        <b>{predicted_conc[0]:.3f} mg/L</b>
    </div>
    """,
    unsafe_allow_html=True
)

    


# ---------------- Tab 1: RGB Analyzer ----------------
with tab1:
    st.header("Average RGB Calculator")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Average RGB
        avg_rgb = img_rgb.mean(axis=(0, 1)).astype(int)
        st.write("Average RGB value:", avg_rgb)

        # Show image and average color side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
        with col2:
            color_patch = np.ones((100, 100, 3), dtype=np.uint8) * avg_rgb
            st.image(color_patch, caption=f"Avg Color {avg_rgb.tolist()}", use_container_width=False)
