'''
quelle: https://www.assemblyai.com/blog/pytorch-lightning-for-dummies/
        https://github.com/aladdinpersson/Machine-Learning-Collection
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # um in Tensorboard zu schreiben


class Diskriminator(nn.Module):
    def __init__(self, eingangs_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(eingangs_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, bild_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, bild_dim),
            nn.Tanh(),  # Normalisierung der Eingaben auf [-1, 1], daher Ausgaben auch [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameter
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 50

diskriminator = Diskriminator(image_dim).to(device)
generator = Generator(z_dim, image_dim).to(device)
festes_rauschen = torch.randn((batch_size, z_dim)).to(device)
transformationen = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

datensatz = datasets.MNIST(root="datensatz/", transform=transformationen, download=True)
dataloader = DataLoader(datensatz, batch_size=batch_size, shuffle=True)
opt_diskriminator = optim.Adam(diskriminator.parameters(), lr=lr)
opt_generator = optim.Adam(generator.parameters(), lr=lr)
verlustfunktion = nn.BCELoss()
schreiber_fake = SummaryWriter(f"logs/fake")
schreiber_real = SummaryWriter(f"logs/real")
schritt = 0

for epoche in range(num_epochs):
    for batch_idx, (echt, _) in enumerate(dataloader):
        echt = echt.view(-1, 784).to(device)
        batch_size = echt.shape[0]

        
        rauschen = torch.randn(batch_size, z_dim).to(device)
        fake = generator(rauschen)
        diskriminator_echt = diskriminator(echt).view(-1)
        verlustD_echt = verlustfunktion(diskriminator_echt, torch.ones_like(diskriminator_echt))
        diskriminator_fake = diskriminator(fake).view(-1)
        verlustD_fake = verlustfunktion(diskriminator_fake, torch.zeros_like(diskriminator_fake))
        verlustD = (verlustD_echt + verlustD_fake) / 2
        diskriminator.zero_grad()
        verlustD.backward(retain_graph=True)
        opt_diskriminator.step()

       
        ausgabe = diskriminator(fake).view(-1)
        verlustG = verlustfunktion(ausgabe, torch.ones_like(ausgabe))
        generator.zero_grad()
        verlustG.backward()
        opt_generator.step()

        if batch_idx == 0:
            print(
                f"Epoche [{epoche}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                      Verlust D: {verlustD:.4f}, Verlust G: {verlustG:.4f}"
            )

            with torch.no_grad():
                fake = generator(festes_rauschen).reshape(-1, 1, 28, 28)
                daten = echt.reshape(-1, 1, 28, 28)
                bild_gitter_fake = torchvision.utils.make_grid(fake, normalize=True)
                bild_gitter_echt = torchvision.utils.make_grid(daten, normalize=True)

                schreiber_fake.add_image(
                    "MNIST Fake Bilder", bild_gitter_fake, global_step=schritt
                )
                schreiber_real.add_image(
                    "MNIST Echte Bilder", bild_gitter_echt, global_step=schritt
                )
                schritt += 1
