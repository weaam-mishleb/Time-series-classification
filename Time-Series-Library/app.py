import streamlit as st
import os
import time
from datetime import timedelta

# --- import your own functions ---
import TimeSeries_functions as ts
from models.TimesNet import Model as TimesNet
from models.Nonstationary_Transformer import Model as NST
from models.Informer import Model as Informer
from models.Autoformer import Model as Autoformer
from models.FEDformer import Model as FEDformer
from models.TimeMixer import Model as Timemixer

MODELS = {
    "moment": None,  # Will handle separately if needed
    "nst": NST,
    "informer": Informer,
    "timesnet": TimesNet,
    "autoformer": Autoformer,
    "fedformer": FEDformer,
    "timemixer": Timemixer,
}

st.set_page_config(page_title="Model Training Panel", layout="wide")

st.title("🧠 Model Training Control Panel")

# --- User inputs ---
dataset_name = st.selectbox("Select Dataset", ["Cesnet", "QuicText","VisQuic","UTMobileNet"])

big_windows = []
if dataset_name in ["Cesnet", "UTMobileNet"]:
    big_windows = ["5s"]  # Fixed 5 seconds for Cesnet
    st.write("Big Window size is fixed at 5 seconds for this dataset")
elif dataset_name == "QuicText":
    big_windows = st.multiselect(
        "Select Big Windows (s)", ["1s","5s","10s"]
    )
else: #  "VisQuic":
    big_windows = ["1s"]  # Fixed 5 seconds for Cesnet
    st.write("Big Window size is fixed at 1 seconds for VisQuic dataset")


small_windows = st.multiselect(
    "Select Small Windows (ms)", ["5ms", "10ms","20ms","30ms","40ms", "50ms","75ms", "100ms","150ms", "200ms", "250ms"]
)

models_selected = st.multiselect(
    "Select Models",
    list(MODELS.keys()),
)

class_counts = []
if dataset_name == "Cesnet":
    class_counts = ["18 classes"]
    st.write("Class count is fixed at 18 for Cesnet dataset")
elif dataset_name == "UTMobileNet":
    class_counts = ["15 classes"]
    st.write("Class count is fixed at 15 for UTMobileNet dataset")
elif dataset_name == "QuicText":
    class_counts = st.multiselect("Select Class Counts", ["3 classes", "4 classes", "5 classes"])
else:  # VisQuic
    class_counts = ["3 classes"]
    st.write("Class count is fixed at 3 for VisQuic dataset")

pos = 2
feature_name="PacketAmount"
selected_feature_indices = []
if dataset_name in [ "QuicText","VisQuic","UTMobileNet"]:
    feature_name = st.selectbox("Select Feature parameter", ["TotalSize", "PacketAmount"])
    pos = 1 if feature_name == "TotalSize" else 2
else:
    feature_name = st.multiselect("Select Feature parameter", ["totalSize_U","PacketAmount_U", "AvgPayload_U",
                                                               "totalSize_D","PacketAmount_D", "AvgPayload_D",
                                                               "ratio"])
    for i, feature in enumerate(feature_name):
        if feature == "totalSize_U":
            selected_feature_indices.append(0)
        elif feature == "PacketAmount_U":
            selected_feature_indices.append(1)
        elif feature == "AvgPayload_U":
            selected_feature_indices.append(2)
        elif feature == "totalSize_D":
            selected_feature_indices.append(3)
        elif feature == "PacketAmount_D":
            selected_feature_indices.append(4)
        elif feature == "AvgPayload_D":
            selected_feature_indices.append(5)
        elif feature == "ratio":
            selected_feature_indices.append(6)
    
load_existing = st.checkbox("Load existing model checkpoint?")
checkpoint_path = None
if load_existing:
    checkpoint_path = st.text_input("Enter checkpoint file path")

run_training = st.button("🚀 Start Training")

if run_training:
    st.success("Starting the training process...")
    datasets_dict = ts.prepare_datasets(dataset_name=dataset_name,
                                    small_windows=small_windows,
                                    big_windows=big_windows,
                                    class_counts=class_counts,
                                    selected_feature_indices=selected_feature_indices,
                                    pos=pos,
                                    )
    st.write("Datasets prepared successfully!")
    results_1 = ts.evaluate_models_on_datasets(datasets_dict=datasets_dict,
                                                model_types=models_selected,
                                                use_class_weights=False,
                                                dataset_name=dataset_name,
                                                feature_name=feature_name
                                                )
    

    st.balloons()
