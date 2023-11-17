import urllib.request
import pickle



import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split


def download_images():
    archivo_local = 'icons'
    if 'icons' not in locals():
        url_dropbox = 'https://www.dropbox.com/s/jgjuf2t0enioz1n/iconos_train.pkl?dl=1'
        # Descargar el archivo
        urllib.request.urlretrieve(url_dropbox, archivo_local)

    # Cargar el diccionario desde el archivo
    with open(archivo_local, 'rb') as handle:
        icons_dict = pickle.load(handle)

    return icons_dict


def creating_loaders(icons_dict):
    images_dataset = torch.tensor(icons_dict['images'], dtype=torch.float32)
    labels = torch.tensor(icons_dict['labels'], dtype=torch.long)

    dataset = TensorDataset(images_dataset, labels)

    training_rate = 0.8

    training_length = int(training_rate * len(dataset))
    test_length = len(dataset) - training_length

    print(training_length, test_length)

    training_dataset, test_dataset = random_split(dataset, [training_length, test_length])

    training_loader = DataLoader(training_dataset, batch_size=512, shuffle=True)
    testing_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    return training_loader, testing_loader


def show_images(loader):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images = images.numpy()
    # Normaliza los datos para que estén en el rango [0, 1]
    datos_normalized = images / images.max()

    # Crea una cuadrícula de subgráficos
    filas = 8
    columnas = 8

    fig, axs = plt.subplots(filas, columnas, figsize=(12, 12))

    # Asegúrate de que 'datos' y 'etiquetas' contengan tus datos reales

    for i in range(filas):
        for j in range(columnas):
            # Obtiene la imagen y la etiqueta correspondiente
            indice = i * columnas + j
            imagen = np.rot90(datos_normalized[indice], k=-1, axes=(1, 2))  # Rota 90 grados a la derecha
            imagen = imagen.transpose((1, 2, 0))  # Transpone para que los canales estén en la última dimensión

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

        # monitor training loss
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        for data, target in train_loader:
            # inicializa los gradientes de todas las variables que optimizamos
            optimizer.zero_grad()

            # paso forward: predice etiquetas de los datos de entrenamiento
            output = model(data)

            # calcula loss
            loss = criterion(output, target)

            # backward pass: calcula gradiente de la loss con respecto parámetros del modelo
            loss.backward()

            # actualiza parametros
            optimizer.step()

            # actualiza loss
            train_loss += loss.item()*data.size(0)

        # calcula la loss media para un epoch entero
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
        
        # comparar predicciones vs. ground truth
        print('Prediction  : {}'.format(pred))
        print('Ground truth: {}'.format(target))

        # compara predicciones con etiquetas ground truth
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        # comprobar los que hemos acertado:
        print(correct)

        # calculate test accuracy for each object class
        for i in range(batch_size):
            # guarda etiqueta ground truth para la muestra actual del batch actual
            label = target.data[i]

            # suma +1 a los aciertos de esta etiqueta, si corrent[i] es True
            class_correct[label] += correct[i].item()

            # aumenta el total de muestras con esta etiqueta
            class_total[label] += 1

# calcula e imprime la loss media
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
        # Capas convolucionales
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Capa totalmente conectada
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Capa de pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Primera capa convolucional con activación ReLU y pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Segunda capa convolucional con activación ReLU y pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Tercera capa convolucional con activación ReLU y pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Aplanar la salida para la capa totalmente conectada
        x = x.view(-1, 128 * 4 * 4)
        
        # Capas totalmente conectadas con activación ReLU
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
if __name__ == '__main__':
    icons = download_images()
    train_loader, test_loader = creating_loaders(icons)
    #show_images(train_loader)
    model = CNN(5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
    model.train()
    train(train_loader=train_loader,criterion=criterion,n_epochs=75)
    test()
    
    
        
    
