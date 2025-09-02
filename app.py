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
tab1, tab2 = st.tabs(["🔎 LOOCV Random Forest", "🎨 RGB Image Analyzer"])

# ---------------- Tab 1: LOOCV ----------------
with tab1:
    st.header("Leave-One-Out Cross Validation (RandomForest)")

    # Example data
    X = np.array([
        [254,254,6],
        [254,254,52],
        [254,254,87],
        [254,253,3],
        [254,253,86],
    ])
    y = np.array([0.007981132, 0.00607177, 0.006714286, 0.008605634, 0.00607177])

    model = RandomForestRegressor(n_estimators=200, random_state=42)

    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)

    st.subheader("Predicted vs Actual (LOOCV)")
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
    r = st.number_input("R value", value=254)
    g = st.number_input("G value", value=254)
    b = st.number_input("B value", value=64)

    new_rgb = np.array([[r, g, b]])
    model.fit(X, y)
    predicted_conc = model.predict(new_rgb)

    st.success(f"Predicted concentration for RGB {new_rgb.tolist()}: {predicted_conc[0]:.6f}")


# ---------------- Tab 2: RGB Analyzer ----------------
with tab2:
    st.header("Average RGB Calculator from Image")

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
