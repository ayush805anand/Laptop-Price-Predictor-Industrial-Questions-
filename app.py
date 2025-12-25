import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Laptop Price Predictor",
    layout="centered"
)

st.image(
    "laptop.jpg",
    use_container_width=True
)

st.title("Laptop Price Prediction")
st.write("Machine learning–based laptop price estimator")

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("onehot_encoder.pkl", "rb") as f:
    onehot = pickle.load(f)

with open("mlb_transformers.pkl", "rb") as f:
    mlb_transformers = pickle.load(f)

with open("feature_metadata.pkl", "rb") as f:
    meta = pickle.load(f)

numeric_features = meta["numeric_features"]
scalar_categorical_features = meta["scalar_categorical_features"]
list_features = meta["list_features"]

CPU_CORE_OPTIONS = sorted(meta["cpu_core_options"])
MEMORY_SIZE_OPTIONS = sorted(
    int(x) for x in meta["memory_size_options"]
    if abs(x - round(x)) < 1e-6 and int(round(x)) != 0
)

st.header("Hardware Configuration")

cpu_cores = st.selectbox(
    "CPU cores",
    CPU_CORE_OPTIONS,
    index=CPU_CORE_OPTIONS.index("4") if "4" in CPU_CORE_OPTIONS else 0
)
st.caption("Select 'not applicable' only if CPU core information is unavailable.")

storage_gb = st.selectbox(
    "Storage size (GB)",
    MEMORY_SIZE_OPTIONS,
    index=MEMORY_SIZE_OPTIONS.index(512) if 512 in MEMORY_SIZE_OPTIONS else 0
)

st.header("Connectivity & I/O")

communications = st.multiselect(
    "Communications",
    options=mlb_transformers["communications"].classes_.tolist()
)

multimedia = st.multiselect(
    "Multimedia",
    options=mlb_transformers["multimedia"].classes_.tolist()
)

input_devices = st.multiselect(
    "Input devices",
    options=mlb_transformers["input devices"].classes_.tolist()
)


if st.button("Predict Price"):
    
    num_row = []
    for col in numeric_features:
        if col == "drive memory size (GB)":
            num_row.append(float(storage_gb))
        else:
            num_row.append(0.0) 

    X_num = scaler.transform([num_row])

    cat_row = []
    for col in scalar_categorical_features:
        if col == "CPU cores":
            cat_row.append(cpu_cores)
        else:
            cat_row.append("unknown")

    X_cat = onehot.transform(
        pd.DataFrame([cat_row], columns=scalar_categorical_features)
    )

    list_blocks = []
    list_inputs = {
        "communications": communications,
        "multimedia": multimedia,
        "input devices": input_devices
    }

    for col in list_features:
        mlb = mlb_transformers[col]
        list_blocks.append(mlb.transform([list_inputs[col]]))

    X_list = np.hstack(list_blocks)

    X_final = np.hstack([X_num, X_cat, X_list])
    predicted_price = model.predict(X_final)[0]

    st.success(f"Predicted Laptop Price: {predicted_price:.2f}")
