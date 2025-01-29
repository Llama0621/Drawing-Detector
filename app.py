import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import torchvision.transforms as transforms
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.ERROR)

if 'initialized' not in st.session_state:
    st.session_state.initialized = True

@st.cache_resource
def load_model(device, num_classes):
    model = EnhancedQuickDrawNet(num_classes=num_classes).to(device)
    try:
        model.load_state_dict(torch.load('best_quickdraw_model.pth', map_location=device))
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is trained first.")
    return model

class EnhancedQuickDrawNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout2d = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout2d(x)
        
        x = x.view(-1, 256 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def process_drawing(image_data, transform):
    img = Image.fromarray(image_data.astype('uint8'))
    img = img.convert('L').resize((28, 28))
    return img, transform(img).unsqueeze(0)

def main():
    # Set page config
    st.set_page_config(
        page_title="Doodle Recognition",
        page_icon="✏️",
        layout="wide"
    )

    st.title("Doodle Recognition")
    st.write("Draw something and let the AI guess what it is!")

    # Sidebar controls
    with st.sidebar:
        st.header("Drawing Controls")
        stroke_width = st.slider("Stroke width: ", 1, 25, 3)
        stroke_color = st.color_picker("Stroke color: ", "#000000")
        bg_color = st.color_picker("Background color: ", "#FFFFFF")

        st.header("Categories")
        categories = ['cat', 'dog', 'fish', 'bird', 'house']
        st.write("Currently trained on: " + ", ".join(categories))

    # Create canvas for drawing
    col1, col2 = st.columns([2, 1])
    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    # Prediction section
    if canvas_result.image_data is not None:
        if st.button("Predict Drawing"):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = load_model(device, len(categories))
                model.eval()

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])

                # Process the drawing
                img, img_tensor = process_drawing(canvas_result.image_data, transform)
                img_tensor = img_tensor.to(device)

                with col2:
                    st.write("Processed Image:")
                    st.image(img, width=150)

                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probabilities = F.softmax(outputs, dim=1)[0]
                        predicted_idx = torch.argmax(probabilities).item()

                        st.write("**Prediction:**")
                        st.write(f"I think this is a **{categories[predicted_idx]}**!")
                        st.write(f"Confidence: {probabilities[predicted_idx]*100:.2f}%")

                        # Show all probabilities
                        st.write("\nProbabilities for all categories:")
                        for idx, prob in enumerate(probabilities):
                            st.write(f"{categories[idx]}: {prob*100:.2f}%")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Instructions
    with st.expander("How to use"):
        st.write("""
        1. Use the controls in the sidebar to adjust your drawing settings
        2. Draw your doodle in the canvas above
        3. Click 'Predict Drawing' to see what the AI thinks you drew
        4. The AI will show its prediction and confidence levels for all possible classes
        """)

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and PyTorch")

if __name__ == "__main__":
    main()