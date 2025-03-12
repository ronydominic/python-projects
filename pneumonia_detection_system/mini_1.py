import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch import optim
from torchvision.models import resnet50
import torch.nn.functional as F
from torchvision import models


device='cpu'
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])



window = tk.Tk()
window.title("Image Classification")
window.minsize(800,800)



class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ImageClassificationModel, self).__init__()
        self.densenet169 = models.densenet169(pretrained=True)
        in_features = self.densenet169.classifier.in_features
        self.densenet169.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes), 
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.densenet169(x)

model = ImageClassificationModel().to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adamax(model.parameters(), lr=0.001)  
model.load_state_dict(torch.load('model_11.pth',map_location=torch.device('cpu')))


# Create labels to display prediction result
result_label = tk.Label(window, text="", font=("Times new roman", 16))
result_label.pack(pady=10)

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        
        image = Image.open(file_path).convert("RGB") 
        image_display = transform_test_display(image)
        tk_image = ImageTk.PhotoImage(image_display)

       
        image_label.config(image=tk_image)
        image_label.image = tk_image

        
        image = transform_test(image)  
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            model.eval()
            output = model(image)
            _, predicted = torch.max(output, 1)

        # Display the predicted class (modify as needed based on your classes)
        class_name = "Normal" if predicted.item() == 0 else "Abnormal"
        result_label.config(text=f"Predicted Class: {class_name}")


# Function to display the selected image
def display_image(file_path):
    image = Image.open(file_path)
    image.thumbnail((300, 300))
    tk_image = ImageTk.PhotoImage(image)

    # Update the label to display the image
    image_label.config(image=tk_image)
    image_label.image = tk_image

# Data Augmentation for displaying images
transform_test_display = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

# Create a button to open the file dialog
open_button = tk.Button(window, text="Open Image", command=open_image)
open_button.pack(pady=10)

# Create a label to display the selected image
image_label = tk.Label(window)
image_label.pack()

# Run the Tkinter event loop
window.mainloop()

