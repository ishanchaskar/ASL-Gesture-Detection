import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import timm
import torch.nn as nn

# Load the class labels (ASL alphabet in your case)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Define the same transformations used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the hybrid model
class EfficientNetDeiTHybrid(nn.Module):
    def _init_(self, num_classes=26):
        super(EfficientNetDeiTHybrid, self)._init_()

        # EfficientNet
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        effnet_out_features = self.efficientnet.num_features

        # DeiT
        self.deit = timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=0)
        deit_out_features = self.deit.embed_dim

        # Final classifier
        self.fc = nn.Sequential(
            nn.Linear(effnet_out_features + deit_out_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        effnet_features = self.efficientnet(x)
        deit_features = self.deit(x)
        combined_features = torch.cat((effnet_features, deit_features), dim=1)
        output = self.fc(combined_features)
        return output

# Load the model
model = EfficientNetDeiTHybrid(num_classes=26)
model.load_state_dict(torch.load('/kaggle/working/efficientnet_deit_best.pth'))
model.eval()
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Function to predict sign from frame
def predict_sign(frame, model, transform, device):
    # Convert the frame (OpenCV image) to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply transformations
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)

    # Get the predicted class
    predicted_class = class_names[preds.item()]
    return predicted_class

# Start real-time webcam capture using OpenCV
def run_realtime_detection(model):
    # Initialize webcam (0 is the default webcam index)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Loop to capture frames from the webcam
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Predict the ASL sign from the frame
        predicted_sign = predict_sign(frame, model, transform, device)

        # Display the result on the frame
        cv2.putText(frame, f'Predicted Sign: {predicted_sign}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('ASL Sign Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the real-time detection
run_realtime_detection(model)