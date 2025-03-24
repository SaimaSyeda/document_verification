# create_placeholder_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import os

class DocumentClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(DocumentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 36 * 36, 128)  # Adjust size based on input
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# Create output directory if it doesn't exist
os.makedirs('models/trained', exist_ok=True)

# Create a simple placeholder model for document classification
model = DocumentClassifier()
torch.save(model.state_dict(), 'models/trained/document_classifier.pth')

# Create a simple placeholder for forgery detector
class SimpleForgeryDetector:
    def predict(self, features):
        return 0  # Always predict "authentic" for testing
    
    def extract_features(self, image):
        return np.zeros(33)  # Return dummy features

detector = SimpleForgeryDetector()
with open('models/trained/forgery_detector.pkl', 'wb') as f:
    pickle.dump(detector, f)

print("Created placeholder models for testing")