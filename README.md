# 🌱 Plant Disease Detection System for Sustainable Agriculture  

### 📌 Overview  
This project leverages **ensemble learning** to improve the accuracy of plant disease detection using deep learning models. The system is designed to provide **real-time disease classification** and is built using **TensorFlow, OpenCV, and Streamlit** for an interactive user experience.  

### 🎯 Goals  
- Develop an **accessible** and **accurate** plant disease detection system.  
- Use **deep learning models** (CNN, ResNet, Inception-ResNet, VGG) for classification.  
- Apply **ensemble learning** (majority voting) to enhance prediction accuracy.  
- Deploy a **real-time** and **user-friendly** web application for farmers.  

---

## 📂 Project Structure  
```plaintext
├── front_end_file.py                        # Streamlit-based frontend for disease detection
├── Plant_Disease_Week_1_CheckPoint.ipynb    # Jupyter Notebook with model training code
├── cnn_model.keras                          # Trained CNN model
├── resnet_model.keras                       # Trained ResNet model
├── inception_resnet_model.keras             # Trained Inception-ResNet model
├── vgg_model.keras                          # Trained VGG model
├── requirements.txt                         # List of dependencies
├── output.png                               # Sample output image
├── README.md                                # Project documentation
```


---

## ⚙️ Technologies Used  
- **Programming Language**: Python  
- **Frameworks & Libraries**:  
  - `TensorFlow`, `Keras` – Model training and deep learning  
  - `OpenCV`, `NumPy` – Image processing  
  - `Scipy` – Statistical computations for ensemble learning  
  - `Streamlit` – Web application deployment  
- **Dataset**: [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  
- **Platform**: Google Colab, Local Machine, Cloud Deployment  

---

## 🚀 How to Run the Project  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/AkshayaGopalakrishnan/Plant_disease_Prediction.git
cd plant-disease-detection
```
2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Run the Streamlit App
```bash
streamlit run front_end_file.py
```

🧠 Model Architecture
The system utilizes an ensemble of deep learning models:

CNN (Convolutional Neural Network)
ResNet (Residual Networks)
Inception-ResNet (Hybrid model)
VGG (Visual Geometry Group model)
Ensemble learning is applied using majority voting, where multiple models predict the disease, and the most common prediction is considered the final output.

📸 Sample Output
Below is an example of how the system predicts plant diseases:


📊 Sample Predictions
The system classifies plant diseases into multiple categories, such as:

Apple Scab, Black Rot, Healthy Apple, etc.
Tomato Early Blight, Tomato Mosaic Virus, etc.

🎯 Future Enhancements
Expand dataset for better generalization.
Integrate mobile app for easier accessibility.
Improve model optimization using pruning and quantization.
