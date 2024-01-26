import streamlit as st
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image

# Load the training data
df = pd.read_csv("D:\\APPLE\\train_data.csv")

# Separate the training dataset
train_label = df["label"]
train_data = df.drop("label", axis=1)

# SVM model
svm_model = svm.SVC(kernel='rbf')
svm_model.fit(train_data, train_label)

# Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(train_data, train_label)

# Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(train_data, train_label)

# Function to process a single image
def process_single_image(image_path):
    # Open the image
    img = Image.open(image_path)

    # Resize the image to 28x28 pixels
    img_resized = img.resize((28, 28))

    # Convert the image to a grayscale array
    img_array = np.array(img_resized.convert("L"))

    # Flatten the array and create a DataFrame for the current image
    img_flat_array = img_array.flatten().reshape(1, -1)

    return img_flat_array

# Function to display confusion matrix
def display_confusion_matrix(true_labels, predictions, model_name):
    # Display confusion matrix
    st.write(f"**Confusion Matrix for {model_name}:**")
    st.write(confusion_matrix(true_labels, predictions))

# Streamlit dashboard with styling
st.set_page_config(
    page_title="Machine Learning Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Custom styling for the title
st.markdown(
    """
    <style>
        .title {
            font-size: 40px;
            color: #3498db;
            text-align: center;
            margin-bottom: 20px;
        }
        .sub-title {
            font-size: 24px;
            color: #555555;
            text-align: center;
            margin-bottom: 20px;
        }
        .image-caption {
            text-align: center;
            margin-top: 10px;
            font-size: 16px;
        }
        .prediction-label {
            text-align: center;
            margin-top: 20px;
            font-size: 24px;
            color: #27ae60;
        }
        .button-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .model-button {
            background-color: #3498db;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            margin: 0 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .model-button:hover {
            background-color: #2980b9;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with additional information
st.sidebar.markdown("<h2 style='color: #3498db;'>Model Information</h2>", unsafe_allow_html=True)
st.sidebar.text("Support Vector Machine (SVM)")
st.sidebar.text("Random Forest Classifier")
st.sidebar.text("Decision Tree Classifier")

# Main content area
st.title("Machine Learning Dashboard")
st.markdown("<h3 class='sub-title'>Image Classification and Model Evaluation</h3>", unsafe_allow_html=True)

# File uploader for image selection with styling
uploaded_file = st.file_uploader("Choose an image...", type="jpg", key="image_uploader")

# Check if a file is uploaded
if uploaded_file is not None:

    # Display the selected image with reduced size
    image = Image.open(uploaded_file)
    resized_image = image.resize((150, 150))
    st.image(resized_image, caption="Uploaded Image", use_column_width=False, output_format="JPEG")

    # Add a class to the image using inline CSS
    st.markdown(
        f'<style>.image-caption img {{max-width: 100%; height: auto;}}'
        '</style>',
        unsafe_allow_html=True
    )

    # Process the uploaded image
    image_data = process_single_image(uploaded_file)

    # Make predictions using the models
    svm_prediction = svm_model.predict(image_data)
    rf_prediction = rf_model.predict(image_data)
    dt_prediction = dt_model.predict(image_data)

    # Check if all three models predict the same label and it does not match the expected label for classification
    if (svm_prediction[0] == rf_prediction[0] == dt_prediction[0]) and (
            svm_prediction[0] != 'this image classification machine'):
        st.markdown("<h2 class='prediction-label'>Image is not related to apple or orange classification.</h2>",
                    unsafe_allow_html=True)
    else:
        # Visual separator
        st.markdown("<hr style='border: 2px solid #3498db;'>", unsafe_allow_html=True)

        # Display predictions with styling
        st.markdown("<h2 class='prediction-label'>SVM Model Prediction:</h2>", unsafe_allow_html=True)
        st.write(f"Predicted Label: {svm_prediction[0]}")

        st.markdown("<h2 class='prediction-label'>Random Forest Model Prediction:</h2>", unsafe_allow_html=True)
        st.write(f"Predicted Label: {rf_prediction[0]}")

        st.markdown("<h2 class='prediction-label'>Decision Tree Model Prediction:</h2>", unsafe_allow_html=True)
        st.write(f"Predicted Label: {dt_prediction[0]}")

        # Visual separator
        st.markdown("<hr style='border: 2px solid #3498db;'>", unsafe_allow_html=True)

        # Display confusion matrix for each model
        display_confusion_matrix([0], [svm_prediction], "SVM")
        display_confusion_matrix([0], [rf_prediction], "Random Forest")
        display_confusion_matrix([0], [dt_prediction], "Decision Tree")

        svm_accuracy = svm_model.score(train_data, train_label)
        rf_accuracy = rf_model.score(train_data, train_label)
        dt_accuracy = dt_model.score(train_data, train_label)

        # Display accuracy on the dashboard
        st.markdown("<h2 class='sub-title'>Model Accuracy:</h2>", unsafe_allow_html=True)
        st.write(f"SVM Accuracy: {svm_accuracy:.2%}")
        st.write(f"Random Forest Accuracy: {rf_accuracy:.2%}")
        st.write(f"Decision Tree Accuracy: {dt_accuracy:.2%}")
