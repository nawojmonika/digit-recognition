import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import torch

model = torch.load('./mnist_model.pt')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
valset = datasets.MNIST('', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

images, labels = next(iter(valloader))

img = images[0].view(1, 784)

with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
image = img.view(1, 28, 28).numpy().squeeze()
plt.imshow(image)
plt.show()
