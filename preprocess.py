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

def preprocess_for_training_with_scaling(dataset):
    url_features_list = dataset['URL'].apply(extract_url_features_from_input)
    features_df = pd.DataFrame(url_features_list.tolist())
    tld_counts = features_df['TLD'].value_counts()
    features_df['TLD_encoded'] = features_df['TLD'].map(tld_counts)
    features_df.drop(['TLD'], axis=1, inplace=True)
    features_df.drop(columns=['NoOfOtherSpecialCharsInURL'], inplace=True)
    dataset_processed = pd.concat([features_df, dataset['label']], axis=1)
    features = dataset_processed.drop('label', axis=1)
    labels = dataset_processed['label']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_dataset = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_dataset['label'] = labels.reset_index(drop=True)
    return scaler, tld_counts



dataset = pd.read_csv('./data.csv')
scaler, tld_counts = preprocess_for_training_with_scaling(dataset)
joblib.dump(scaler,"scaler.joblib")
joblib.dump(tld_counts,"tld_counts.joblib")




