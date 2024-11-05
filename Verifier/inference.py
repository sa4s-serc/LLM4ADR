from matplotlib import lines
import torch
import torch.nn as nn
import os
import google.generativeai as genai
from openai import OpenAI
from nltk.tokenize import word_tokenize
import sys
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import json

class BaseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(BaseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 2048)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(2048, 2048)  

    def forward(self, x):
        return x

# Define the Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        self.base_network_c = BaseNetwork(input_dim)
        self.base_network_d = BaseNetwork(input_dim)
        
        # Additional hidden layers after concatenation
        self.hidden1 = nn.Linear(input_dim * 2, 2048)
        self.dropout1 = nn.Dropout(0.0)
        self.hidden2 = nn.Linear(2048, 2048)
        self.dropout2 = nn.Dropout(0.0)
        self.hidden3 = nn.Linear(2048, 2048)
        self.dropout3 = nn.Dropout(0.0)
        self.classifier = nn.Linear(2048, 2)  
        
    def forward_once_c(self, x):
        return self.base_network_c(x)
    
    def forward_once_d(self, x):
        return self.base_network_d(x)

    def forward(self, context, decision):
        context_output = self.forward_once_c(context)
        decision_output = self.forward_once_d(decision)
        combined_output = torch.cat((context_output, decision_output), dim=1)
        
        x = torch.relu(self.hidden1(combined_output))
        x = self.dropout1(x)
        x = torch.relu(self.hidden2(x))
        x = self.dropout2(x)
        x = torch.relu(self.hidden3(x))
        x = self.dropout3(x)
        logits = self.classifier(x)
        return logits
    
# Load the environment variables
load_dotenv(find_dotenv(raise_error_if_not_found=True))

if os.environ.get("your_api_key") is None or os.environ.get("OPENAI_API_KEY") is None:
    raise ValueError("Please set your API keys in the environment variables")

genai.configure(api_key=os.environ['your_api_key'])
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# A helper function to get the OpenAI embeddings
def get_openai_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    embeddings = [embedding.embedding for embedding in response.data]
    single_list_embeddings = [embedding for sublist in embeddings for embedding in sublist]
    return single_list_embeddings

# Function to get embeddings with the selected model
def get_embeddings(context, decision, model='gemini'):
    max_tokens = 2048
    
    # Tokenize and count tokens
    context_tokens = word_tokenize(context)
    decision_tokens = word_tokenize(decision)
    
    if len(context_tokens) > max_tokens or len(decision_tokens) > max_tokens:
        return None, None, True  # Return None embeddings and error flag if tokens exceed max_tokens
    
    if model == 'gemini':
        context_embedding_dict = genai.embed_content(model="models/text-embedding-004", content=context)
        decision_embedding_dict = genai.embed_content(model="models/text-embedding-004", content=decision)
        context_embedding = torch.tensor(context_embedding_dict['embedding'])
        decision_embedding = torch.tensor(decision_embedding_dict['embedding'])
    if model == 'openai':
        context_embedding = torch.tensor(get_openai_embedding(context, model='text-embedding-3-large'))
        decision_embedding = torch.tensor(get_openai_embedding(decision, model='text-embedding-3-large'))
        # print(context_embedding.shape, decision_embedding.shape)

    return context_embedding, decision_embedding, False

# Function to load a saved model from a pkl file from the given path
def load_model(model_path, input_dim):
    model = SiameseNetwork(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

# Function to infer the class and confidence from the selected loaded model with given inputs as both context and decision embeddings
def get_prediction(model, context_embedding, decision_embedding):
    # X_combined = torch.cat((context_embedding, decision_embedding), dim=0).unsqueeze(0)

    with torch.no_grad():
        outputs = model(context_embedding.unsqueeze(0), decision_embedding.unsqueeze(0))
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)

    predicted_class = predicted.item()
    confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence

# Load both models at startup
model_paths = {
    'gemini': './models/gemini.pth',
    'openai': './models/openai.pth'
}
google_model = load_model(model_paths['gemini'], input_dim=768)
openai_model = load_model(model_paths['openai'], input_dim=3072)

def predict(context, decision, model_name='gemini'):
    context_embedding, decision_embedding, error = get_embeddings(context, decision, model_name)

    if error:
        raise ValueError('Context or Decision length exceeds 999 characters.')

    # Use the pre-loaded model based on the request
    model = google_model if model_name == 'gemini' else openai_model
    predicted_class, confidence = get_prediction(model, context_embedding, decision_embedding)

    response = {
        'prediction': predicted_class,
        'probability': confidence
    }

    return response

def write_jsonl(filename, data):
    with open(filename, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def main():
    file_name = sys.argv[1]
    model_name = sys.argv[2]
    data = pd.read_json(file_name, lines = True)
    
    correct = 0
    avg_confidence = 0
    responses = []
    
    for index, row in tqdm(data.iterrows(), total=len(data)):
        context = row['Context']
        decision = row['Predictions']
    
        response = predict(context, decision, model_name)
        response['id'] = row['id']
        responses.append(response)
        correct += response['prediction']
        
        if response['prediction'] == 1:
            avg_confidence += response['probability']
        else:
            avg_confidence += 1 - response['probability']
            
    accuracy = correct / len(data)
    avg_confidence = avg_confidence / len(data)
    
    responses.insert(0, {'accuracy': accuracy, 'avg_confidence': avg_confidence})
    
    output_filename = f'results/{model_name}/{file_name.split("/")[-1].split(".")[0]}.jsonl'
    write_jsonl(output_filename, responses)
    
    # print(f'Accuracy: {accuracy}')
    # print(f'Average Confidence: {avg_confidence}')

if __name__ == '__main__':
    main()
