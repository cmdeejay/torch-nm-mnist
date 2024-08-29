from torch import nn, save, load
import torch
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# download the data set
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
# load the dataset
dataset = DataLoader(train, 32)
# choose the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# shape of the dataset 1, 28, 28

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10),
        )

    def forward(self, x):
        return self.model(x)

    # Instance of the neural network, loss, optimizer


clf = ImageClassifier().to(device)
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
# if __name__ == "__main__":
    # for epoch in range(10):
    #     for batch in dataset:
    #         X, y = batch
    #         X, y = X.to(device), y.to(device)
    #         yhat = clf(X)
    #         loss = loss_fn(yhat, y)

    #         # apply backprop
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    #     print(f"Epoch: {epoch} loss is {loss.item()}")

    # with open("model_state.pt", "wb") as f:
    #     save(clf.state_dict(), f)

# Prediction
if __name__ == '__main__':
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))
    
    img = Image.open('8.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    print(torch.argmax(clf(img_tensor)))