import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# --------------------- MODEL DEFINITION ---------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --------------------- TRANSFORMS ---------------------
def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

# --------------------- DEVICE ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- APP UI ---------------------
st.title("CIFAR-10 Image Classifier")

# Upload trained model
uploaded_model = st.file_uploader("Upload trained model (.pth)", type=["pth"])
if uploaded_model:
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(uploaded_model, map_location=device))
    model.to(device)
    model.eval()
    st.success("✅ Model loaded successfully!")

# Upload an image
uploaded_file = st.file_uploader("Upload an image to classify", type=["png","jpg","jpeg"])
if uploaded_file and uploaded_model:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess
    transform = get_transforms(augment=False)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        pred_class = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
    
    # Map to CIFAR-10 classes
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    st.write(f"Predicted class: **{classes[pred_class]}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
