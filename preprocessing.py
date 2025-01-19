import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import string

# Below are the functions pertaining to the preprocessing for the dataset.
def extract_url_features_from_input(url):
    """
    Extract features from a single URL input.
    
    Parameters:
    url (str): A single URL to extract features from.
    
    Returns:
    dict: A dictionary containing the extracted features.
    """
    features = {}

    # Feature: URL Length
    features['URLLength'] = len(url)

    # Feature: Number of Subdomains
    domain = url.split("://")[-1].split("/")[0]
    features['NoOfSubDomain'] = domain.count('.') - 1 if '.' in domain else 0

    # Feature: Obfuscation
    special_chars = ["@", "-", "_", "/", "=", "?"]
    features['HasObfuscation'] = 1 if any(char in url for char in special_chars) else 0
    features['NoOfObfuscatedChar'] = sum(char in special_chars for char in url)
    features['ObfuscationRatio'] = features['NoOfObfuscatedChar'] / features['URLLength']

    # Feature: Character Continuation Rate
    features['CharContinuationRate'] = sum(url[i] == url[i - 1] for i in range(1, len(url))) / len(url) if len(url) > 1 else 0

    # Feature: Letter Ratios and Counts
    features['NoOfLettersInURL'] = sum(char.isalpha() for char in url)
    features['LetterRatioInURL'] = features['NoOfLettersInURL'] / features['URLLength']

    # Feature: Special Characters Ratio
    special_counts = ['=', '?', '&', '@', '-']
    features['SpacialCharRatioInURL'] = sum(url.count(char) for char in special_counts) / features['URLLength']

    # Feature: Number of Other Special Characters in URL
    all_special_chars = set(string.punctuation)  # All special characters
    known_special_chars = set(special_chars)    # Already counted special characters
    other_special_chars = all_special_chars - known_special_chars  # Remaining characters
    features['NoOfOtherSpecialCharsInURL'] = sum(char in other_special_chars for char in url)

    # Feature: Is HTTPS
    features['IsHTTPS'] = 1 if url.startswith('https://') else 0

    # Feature: TLD (Last part of domain after '.')
    tld = domain.split('.')[-1]
    features['TLD'] = tld

    return features
    pass

def preprocess_for_training_with_scaling(dataset):
    """
    Preprocess the dataset for training by extracting features, encoding, and scaling.
    Removes unused features like 'NoOfOtherSpecialCharsInURL'.
    """
    # Extract features for each URL in the dataset
    url_features_list = dataset['URL'].apply(extract_url_features_from_input)

    # Convert the list of features into a DataFrame
    features_df = pd.DataFrame(url_features_list.tolist())

    # Frequency encoding for TLD
    tld_counts = features_df['TLD'].value_counts()
    features_df['TLD_encoded'] = features_df['TLD'].map(tld_counts)

    # Drop the original TLD column as it is now encoded
    features_df.drop(['TLD'], axis=1, inplace=True)

    # Remove unused features
    features_df.drop(columns=['NoOfOtherSpecialCharsInURL'], inplace=True)

    # Concatenate the processed features with the original labels
    dataset_processed = pd.concat([features_df, dataset['label']], axis=1)

    # Separate features and labels for scaling
    features = dataset_processed.drop('label', axis=1)
    labels = dataset_processed['label']

    # Initialize scaler and scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Combine scaled features with labels
    scaled_dataset = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_dataset['label'] = labels.reset_index(drop=True)

    return scaled_dataset, scaler, tld_counts
    pass

def preprocess_single_url(url, scaler, tld_counts):
    """
    Preprocess a single URL for prediction using the trained model.
    Removes 'NoOfOtherSpecialCharsInURL' during preprocessing.
    """
    # Extract features from the URL
    features = extract_url_features_from_input(url)

    # Encode TLD using frequency counts
    features['TLD_encoded'] = tld_counts.get(features['TLD'], 0)

    # Drop original TLD key as it is now encoded
    features.pop('TLD', None)

    # Remove unused feature
    features.pop('NoOfOtherSpecialCharsInURL', None)

    # Convert features to a DataFrame
    features_df = pd.DataFrame([features])

    # Scale features using the provided scaler
    scaled_features = scaler.transform(features_df)

    return scaled_features
    pass


# Define the PyTorch model (NNModel)
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

# Load the trained model
device = torch.device('cpu')
model_loaded = torch.load('LinkStop_NNmodel1.pth', map_location=device)
model_loaded.eval()

# Load and preprocess the dataset
dataset = pd.read_csv('phiusiil+phishing+url+dataset\PhiUSIIL_Phishing_URL_Dataset.csv')
scaled_dataset, scaler, tld_counts = preprocess_for_training_with_scaling(dataset)

# Prediction for a single URL
def classify_url(url):
    try:
        # Preprocess the URL
        scaled_features = preprocess_single_url(url, scaler, tld_counts)
        test_row_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)
        
        # Predict with the model
        with torch.no_grad():
            prediction = model_loaded(test_row_tensor)
            predicted_class = (prediction > 0.5).float().item()
        return prediction.item(), int(predicted_class)
    except Exception as e:
        return f"Error: {e}", None

# Example usage
test_url = "https://chatgpt.com/c/6780178e-68ec-8001-a476-62834ec48746"
# test_url = "https://github.com/Raj-Aarav/LinkStop_beta/blob/main/prediction.py"
probability, label = classify_url(test_url)

# Output the result
print(f"URL: {test_url}")
print(f"Predicted Probability: {probability:.4f}")
print(f"Predicted Label: {'Malicious' if label == 1 else 'Safe (Non-Malicious)'}")
