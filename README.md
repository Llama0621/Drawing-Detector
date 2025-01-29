Drawing Detector
A simple web application that allows users to draw and receive real-time AI predictions of their drawings. Built with Streamlit and PyTorch.

Features
Interactive Drawing Canvas: Draw your creations directly on the web interface.
Real-time Predictions: Get instant category predictions for your drawings.
Supported Categories: Cat, Dog, Fish, Bird, and House.
Installation
Prerequisites
Python 3.7 or higher: Download Python
Git: Download Git
Steps
Clone the Repository

bash
Copy
git clone https://github.com/yourusername/drawing-detector.git
cd drawing-detector
Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

bash
Copy
python -m venv venv
Activate the Virtual Environment

Windows

bash
Copy
venv\Scripts\activate
macOS/Linux

bash
Copy
source venv/bin/activate
Install Dependencies

bash
Copy
pip install -r requirements.txt
If requirements.txt is not available, install manually:

bash
Copy
pip install streamlit torch torchvision streamlit-drawable-canvas pillow numpy scikit-learn requests
Usage
Train the Model
Before running the app, you need to train the AI model:

Run the Training Script

bash
Copy
python train.py
What the Script Does:

Data Download: Downloads the QuickDraw dataset for the specified categories.
Data Processing: Preprocesses and augments the data.
Model Training: Trains the EnhancedQuickDrawNet model with the training data.
Validation: Evaluates the model on a validation set and saves the best-performing model as best_quickdraw_model.pth.
Notes:

Training may take some time depending on your hardware.
Ensure you have a stable internet connection to download the dataset.
The model architecture is defined within train.py and matches the one used in app.py.
Verify Model Saving

After successful training, a file named best_quickdraw_model.pth will be present in the project directory. This file is essential for the application to make predictions.

Run the Application
Once the model is trained, launch the web application:

Ensure the Model File Exists

Make sure best_quickdraw_model.pth is in the project directory.

Run the Streamlit App

bash
Copy
streamlit run app.py
Access the Application

After running the above command, Streamlit will provide a local URL (typically http://localhost:8501). Open this URL in your web browser to access the Drawing Detector App.

Project Structure
kotlin
Copy
drawing-detector/
├── app.py
├── train.py
├── best_quickdraw_model.pth
├── data/
│   ├── cat.npy
│   ├── dog.npy
│   ├── fish.npy
│   ├── bird.npy
│   └── house.npy
├── requirements.txt
└── README.md
app.py: Streamlit application for drawing detection.
train.py: Script to train the AI model.
best_quickdraw_model.pth: Trained model file.
data/: Dataset files for each category.
requirements.txt: Python dependencies.
README.md: This documentation file.
Dependencies
The project relies on the following Python libraries:

Streamlit: Web application framework.
PyTorch: Deep learning library for model training and inference.
Torchvision: Utilities for image processing and model handling.
Streamlit Drawable Canvas: Provides the interactive drawing canvas in the app.
Pillow (PIL): Image processing library.
NumPy: Numerical computing.
Scikit-learn: Utilities for data splitting and evaluation.
Requests: HTTP library for data downloading.
