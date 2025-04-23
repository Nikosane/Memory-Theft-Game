import torch
import torch.nn as nn
import torch.optim as optim
from model.tokenizer import SimpleTokenizer

class MemoryThief:
    def __init__(self):
        self.tokenizer = SimpleTokenizer()
        self.model = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 32)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.fc.parameters()), lr=0.001)
        self.criterion = nn.MSELoss()

    def train_step(self, text):
        tokens = self.tokenizer.tokenize(text)
        inputs = torch.tensor(tokens).float().unsqueeze(0)
        targets = inputs.clone()

        self.optimizer.zero_grad()
        outputs, _ = self.model(inputs)
        outputs = self.fc(outputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

    def generate_text(self):
        seed = torch.randn(1, 10, 32)
        output, _ = self.model(seed)
        output = self.fc(output)
        return self.tokenizer.detokenize(output.squeeze(0).tolist())

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'fc_state_dict': self.fc.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.fc.load_state_dict(checkpoint['fc_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

