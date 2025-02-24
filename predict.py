import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import string
import joblib  # Replacing pickle
from sklearn.preprocessing import StandardScaler

def extract_url_features_from_input(url):
    features = {}
    features['URLLength'] = len(url)
    domain = url.split("://")[-1].split("/")[0]
    features['NoOfSubDomain'] = domain.count('.') - 1 if '.' in domain else 0
    special_chars = ["@", "-", "_", "/", "=", "?"]
    features['HasObfuscation'] = 1 if any(char in url for char in special_chars) else 0
    features['NoOfObfuscatedChar'] = sum(char in special_chars for char in url)
    features['ObfuscationRatio'] = features['NoOfObfuscatedChar'] / features['URLLength']
    features['CharContinuationRate'] = sum(url[i] == url[i - 1] for i in range(1, len(url))) / len(url) if len(url) > 1 else 0
    features['NoOfLettersInURL'] = sum(char.isalpha() for char in url)
    features['LetterRatioInURL'] = features['NoOfLettersInURL'] / features['URLLength']
    special_counts = ['=', '?', '&', '@', '-']
    features['SpacialCharRatioInURL'] = sum(url.count(char) for char in special_counts) / features['URLLength']
    all_special_chars = set(string.punctuation)
    known_special_chars = set(special_chars)
    other_special_chars = all_special_chars - known_special_chars
    features['NoOfOtherSpecialCharsInURL'] = sum(char in other_special_chars for char in url)
    features['IsHTTPS'] = 1 if url.startswith('https://') else 0
    tld = domain.split('.')[-1]
    features['TLD'] = tld
    return features

def preprocess_single_url(url, scaler, tld_counts):
    features = extract_url_features_from_input(url)
    features['TLD_encoded'] = tld_counts.get(features['TLD'], 0)
    features.pop('TLD', None)
    features.pop('NoOfOtherSpecialCharsInURL', None)
    features_df = pd.DataFrame([features])
    scaled_features = scaler.transform(features_df)
    return scaled_features

class NNModel(nn.Module):
    def __init__(self, input_size):
        super(NNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 5)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

device = torch.device('cpu')
input_size = 11 
model = NNModel(input_size)
model.load_state_dict(torch.load('./model.pth', map_location=device, weights_only=True))
model.eval()

scaler = joblib.load('scaler.joblib')
tld_counts = joblib.load('tld_counts.joblib')

def classify_url(url):
    try:
        scaled_features = preprocess_single_url(url, scaler, tld_counts)
        test_row_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
        with torch.no_grad():
            prediction = model(test_row_tensor)
            predicted_class = (prediction > 0.05).float().item()
        return prediction.item(), int(predicted_class)
    except Exception as e:
        return f"Error: {e}", None

# Example usage
test_url = "https://github.com/Raj-Aarav/LinkStop_beta/blob/main/preprocessing.py"
mal_url = "https://g00gle.tt/@=??"
probability, label = classify_url(test_url)

print('\n-------------------------------------------------\n')
print(f"URL: {test_url}")
print('\n-------------------------------------------------\n')
print(f"Predicted Probability: {probability:.4f}")
print(f"Predicted Label: {'Non-Malicious' if label == 1 else 'Malicious'}")
print('\n-------------------------------------------------\n')