# In models/forgery_detector.py
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class ForgeryDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def extract_features(self, image):
        # Edge detection
        edges = cv2.Canny(image, 100, 200)
        
        # Compute histogram of oriented gradients (HOG)
        # Simplified version
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        # Create histogram features
        hist_features = np.histogram(mag.ravel(), bins=32)[0]
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        # Combine features
        features = np.concatenate([hist_features, [edge_density]])
        
        return features
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, image):
        features = self.extract_features(image)
        return self.model.predict([features])[0]