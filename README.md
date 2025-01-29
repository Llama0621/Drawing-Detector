Drawing Detector
A simple web application that allows users to draw doodles and receive real-time AI predictions of their drawings. Built with Streamlit and PyTorch.

Features
Interactive Drawing Canvas: Draw your doodles directly on the web interface.
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
git clone https://github.com/yourusername/doodle-recognition-app.git
cd doodle-recognition-app
Create a Virtual Environment

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
Before running the app, train the AI model:

bash
Copy
python train.py
This will download the necessary data, train the model, and save the best model as best_quickdraw_model.pth.

Run the Application
After training, launch the web app:

bash
Copy
streamlit run app.py
Open the provided local URL (e.g., http://localhost:8501) in your web browser to use the Doodle Recognition App.
