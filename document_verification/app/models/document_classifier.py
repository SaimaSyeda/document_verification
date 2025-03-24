# In models/document_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def create_document_classifier(num_classes=3):
    return DocumentClassifier(num_classes)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, num_classes=3):
    model = DocumentClassifier(num_classes)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model