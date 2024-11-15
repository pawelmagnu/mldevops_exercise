import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: Compress the image
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, 16),
        )

        # Decoder: Reconstruct the image
        self.decoder = nn.Sequential(
            nn.Linear(16, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Flatten the input image
        x = x.view(-1, 28 * 28)
        # Encode and decode
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Reshape back to image format
        decoded = decoded.view(-1, 1, 28, 28)
        return decoded


def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:  # We don't need the labels for autoencoders
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)  # Compare the output (reconstructed) with the input

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(loader):.4f}")


def visualize_reconstruction(model, loader):
    model.eval()
    images, _ = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        reconstructed = model(images)

    # Plot the original and the reconstructed images
    fig, axs = plt.subplots(2, 10, figsize=(12, 4))
    for i in range(10):
        # Original images
        axs[0, i].imshow(images[i].cpu().squeeze(), cmap="gray")
        axs[0, i].set_title("Original")
        axs[0, i].axis("off")

        # Reconstructed images
        axs[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap="gray")
        axs[1, i].set_title("Reconstructed")
        axs[1, i].axis("off")

    plt.show()
