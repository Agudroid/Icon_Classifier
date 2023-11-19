import urllib.request
import pickle



import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, random_split


def download_images():
    archivo_local = 'icons'
    if 'icons' not in locals():
        url_dropbox = 'https://www.dropbox.com/s/jgjuf2t0enioz1n/iconos_train.pkl?dl=1'
        urllib.request.urlretrieve(url_dropbox, archivo_local)

    with open(archivo_local, 'rb') as handle:
        icons_dict = pickle.load(handle)

    return icons_dict


def creating_loaders(icons_dict):
    images_dataset = torch.tensor(icons_dict['images'], dtype=torch.float32)
    labels = torch.tensor(icons_dict['labels'], dtype=torch.long)
    images_dataset, labels = add_noisy_image(images_dataset,labels)

    dataset = TensorDataset(images_dataset, labels)

    training_rate = 0.9

    training_length = int(training_rate * len(dataset))
    test_length = len(dataset) - training_length

    print(training_length, test_length)

    training_dataset, test_dataset = random_split(dataset, [training_length, test_length])

    training_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)
    testing_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return training_loader, testing_loader


def add_noisy_image(images, labels):
    original_images = images

    rotation_transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.RandomRotation(90),
                                            transforms.ToTensor()])

    rotated_images = torch.cat([original_images, torch.stack([rotation_transform(img) for img in original_images])])

    duplicated_labels = torch.cat([labels])
    extended_labels = torch.cat([labels, duplicated_labels])
    print("Forma del tensor original:", images.shape)
    print("Forma del tensor con imágenes rotadas:", rotated_images.shape)
    
    print("Forma del tensor original de etiquetas:", labels.shape)
    print("Forma del tensor extendido de etiquetas:", extended_labels.shape)

    return rotated_images, extended_labels    
    
    
def show_images(loader):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images = images.numpy()
    datos_normalized = images / images.max()

    filas = 8
    columnas = 8

    fig, axs = plt.subplots(filas, columnas, figsize=(12, 12))


    for i in range(filas):
        for j in range(columnas):
            indice = i * columnas + j
            imagen = np.rot90(datos_normalized[indice], k=-1, axes=(1, 2))  
            imagen = imagen.transpose((1, 2, 0)) 

            etiqueta = labels[indice]

            # Muestra la imagen en el subgráfico correspondiente
            axs[i, j].imshow(imagen)
            axs[i, j].set_title(f'Etiqueta: {etiqueta}')
            axs[i, j].axis('off')  # Desactiva los ejes

    plt.tight_layout()
    plt.show()

def train(train_loader, criterion, n_epochs=15):
    losses = []
    for epoch in range(n_epochs):


        train_loss = 0.0


        for data, target in train_loader:

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()*data.size(0)

        train_loss = train_loss/len(train_loader.dataset)

        losses.append(train_loss)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch+1,
            train_loss
            ))

def test():
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    model.eval()
    
    for data, target in test_loader:
        batch_size = data.size(0)
        output = model(data)
        loss = criterion(output,target)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)
        
        print('Prediction  : {}'.format(pred))
        print('Ground truth: {}'.format(target))

        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        print(correct)

        for i in range(batch_size):
            label = target.data[i]

            class_correct[label] += correct[i].item()

            class_total[label] += 1

    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))


    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(1024 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Primera capa convolucional con activación ReLU y pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Segunda capa convolucional con activación ReLU y pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Tercera capa convolucional con activación ReLU y pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Aplanar la salida para la capa totalmente conectada
        x = x.view(-1, 1024 * 4 * 4)
        
        # Capas totalmente conectadas (añadimos la función de activación ReLU ya que es una función no lineal)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
if __name__ == '__main__':
    icons = download_images()
    train_loader, test_loader = creating_loaders(icons)
    #show_images(train_loader)
    model = CNN(5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    model.train()
    train(train_loader=train_loader,criterion=criterion,n_epochs=25)
    test()
    
    
        
    
