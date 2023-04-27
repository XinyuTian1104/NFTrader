import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_size=768, output_size=2048):
        super(BertEncoder, self).__init__()

        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)

        # Define linear layer to transform BERT output to desired output size
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, text):
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors='pt')

        # Pass input through BERT model
        outputs = self.bert(**inputs)

        # Extract the last hidden state of the BERT output
        last_hidden_state = outputs.last_hidden_state

        # Apply linear layer to the last hidden state to get the feature vector
        feature_vector = self.linear(last_hidden_state[:, 0, :])

        return feature_vector


def test_bert():
    # Instantiate BertEncoder module
    model = BertEncoder()
    # print(model)

    # Define random text
    text = "This is a sample text to test the BertEncoder module."

    # Encode text into feature vector with length 2048
    feature_vector = model(text)

    # Print feature vector
    print(feature_vector.size())
