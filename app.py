import streamlit as st
import numpy as np
import librosa
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf

st.set_page_config(page_title="SVM – Spoken Command Classifier", layout="wide")

# ----------------------
# Helper Functions
# ----------------------

def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled


def train_svm(X, y):
    model = SVC(kernel="rbf", probability=True)
    model.fit(X, y)
    return model


def plot_conf_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)


# ----------------------
# Streamlit UI
# ----------------------

st.title("🎤 SVM – Spoken Command Classifier")
st.write("Upload audio files, train an SVM model, and classify new commands.")

# Sidebar navigation
menu = st.sidebar.radio("Menu", ["Train Model", "Predict Command"])


# -----------------------------------------
# TRAINING SECTION
# -----------------------------------------
if menu == "Train Model":

    st.header("📚 Train the SVM Model")
    st.write("Upload audio files organized by class folders:")

    st.info("""
**Folder Format Example:**

- yes/
  - yes1.wav
  - yes2.wav  
- no/
  - no1.wav  
  - no2.wav  

Upload these folders as a zipped file.
    """)

    uploaded_zip = st.file_uploader("Upload dataset ZIP", type=["zip"])

    if uploaded_zip:
        import zipfile
        import tempfile

        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "data.zip")

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        st.success("Dataset extracted!")

        X, y = [], []
        labels = []

        # Scan folders
        for label in os.listdir(temp_dir):
            folder = os.path.join(temp_dir, label)
            if os.path.isdir(folder):
                labels.append(label)
                for file in os.listdir(folder):
                    if file.endswith(".wav"):
                        file_path = os.path.join(folder, file)
                        features = extract_mfcc(file_path)
                        X.append(features)
                        y.append(label)

        X, y = np.array(X), np.array(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write("Training SVM model...")
        model = train_svm(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"Model trained successfully! 🎉")
        st.write(f"**Accuracy:** {acc * 100:.2f}%")

        st.subheader("Confusion Matrix")
        plot_conf_matrix(y_test, y_pred, labels)

        # Save model
        joblib.dump(model, "svm_audio_model.pkl")
        joblib.dump(labels, "labels.pkl")
        st.success("Model saved as `svm_audio_model.pkl` and `labels.pkl`")


# -----------------------------------------
# PREDICTION SECTION
# -----------------------------------------
elif menu == "Predict Command":

    st.header("🎯 Predict Spoken Command")

    # Load saved model
    if not os.path.exists("svm_audio_model.pkl"):
        st.error("⚠ Train a model first in the Training tab!")
    else:
        model = joblib.load("svm_audio_model.pkl")
        labels = joblib.load("labels.pkl")

        option = st.radio("Choose Input Type", ["Upload Audio", "Record Audio"])

        # ----- UPLOAD AUDIO -----
        if option == "Upload Audio":
            file = st.file_uploader("Upload a WAV file", type=["wav"])

            if file:
                with open("temp.wav", "wb") as f:
                    f.write(file.read())

                features = extract_mfcc("temp.wav").reshape(1, -1)

                preds = model.predict_proba(features)[0]
                top_idx = np.argmax(preds)
                st.success(f"🗣 Predicted Command: **{labels[top_idx]}**")

                st.write("Confidence:")
                for lbl, p in zip(labels, preds):
                    st.write(f"- {lbl}: {p*100:.2f}%")

        # ----- RECORD AUDIO (optional) -----
        if option == "Record Audio":
            st.info("Use any online voice recorder to record a 1-sec WAV & upload it above.")
            st.write("Streamlit microphone support can be added if needed.")

