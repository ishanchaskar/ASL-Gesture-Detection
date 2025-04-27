import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image
import math

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model
model_path = 'efficientnet_b0_best.pth'  # Adjust to your model path
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=26)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Define transformations for the test image
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class labels for reference
class_labels = [chr(i) for i in range(65, 91)]  # A=0, Z=25

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Set parameters for cropping the hand image
offset = 20
imgSize = 224

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Ensure crop boundaries are within the image
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        
        imgCrop = img[y1:y2, x1:x2]
        
        if imgCrop.size != 0:  # Check for empty crop
            # Resize to square (224x224)
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Convert to PIL image for model input
            imgRGB = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(imgRGB)
            input_tensor = test_transforms(pil_image).unsqueeze(0).to(device)

            # Perform prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_idx = predicted.item()
                predicted_label = class_labels[predicted_idx]
                print(f"Predicted: {predicted_label}")
                
                # Display prediction on the image, positioned away from the hand area
                text_offset = 40  # Adjust for better positioning
                cv2.putText(img, f'{predicted_label}', (x1, y1 - text_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("ImageWhite", imgWhite)

    # Display the image with hand landmarks
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()