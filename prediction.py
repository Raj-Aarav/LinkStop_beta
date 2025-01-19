import torch
import torch.nn as nn
import pandas as pd

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
model_loaded = torch.load('./LinkStop_NNmodel1.pth', map_location=device)
model_loaded.eval()

#Read the values from the input url and covnvert it into dataframe and then pass it over here , ideally its .csv with only one row for prediction.

dataset = pd.read_csv('./pro_dataset.csv')
# Reomve this line if you have already dropped this column in the updated datafeed
dataset = dataset.drop(columns=['NoOfOtherSpecialCharsInURL'])

# features_to_keep = [
#     'URLLength', 'CharContinuationRate', 'NoOfSubDomain', 'HasObfuscation', 
#     'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfLettersInURL', 
#     'LetterRatioInURL', 'SpacialCharRatioInURL', 
#     'IsHTTPS', 'TLD_encoded', 'label'
# ]


#We have only considered these features

y = dataset['label']
X = dataset.drop('label', axis=1)

test_row = X.iloc[1033].values
true_label = y.iloc[1033]

test_row_tensor = torch.tensor(test_row, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    prediction = model_loaded(test_row_tensor)
    predicted_class = (prediction > 0.5).float().item()

print(f"True Label: {true_label}")
print(f"Predicted Probability: {prediction.item():.4f}")
print(f"Predicted label: {int(predicted_class)}")

print('---------------------------------------------')
if(true_label==predicted_class):
    print('Correct prediction')
else:
    print('Wrong prediction')
print('---------------------------------------------')
