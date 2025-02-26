import os
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from tqdm import tqdm

from models.byol_network import ByolNet
from models.classifier import classifier


def load_weigths(config):
    checkpoint_folder = config['network']['checkpoints']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    byol_model = ByolNet(**config['network']).to(device)

    load_params = torch.load(os.path.join(checkpoint_folder, 'byol_model.pth'),
                             map_location=device)
    byol_model.load_state_dict(load_params['online_network_state_dict'])

    num_classes = 10
    if config['network']['pretrained']:
        in_channels = byol_model.byolnet.heads.head.in_features
    else:
        in_channels = byol_model.byolnet.mlp_head[-1].in_features
    classifier_model = classifier(in_channels=in_channels, num_classes=num_classes).to(device)

    load_params = torch.load(os.path.join(checkpoint_folder, 'classifier_model.pth'),
                             map_location=device)
    classifier_model.load_state_dict(load_params['classifier_state_dict'])

    time_byol = os.path.getmtime(os.path.join(checkpoint_folder, 'byol_model.pth'))
    time_classifier = os.path.getmtime(os.path.join(checkpoint_folder, 'classifier_model.pth'))

    return byol_model, classifier_model, time_byol, time_classifier


def classifier_inference(test_data, byol, classifier, byol_time, classifier_time, sigma=0.1, batch_size=256,
                         noisy_inference=False, write_to_log=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
    )

    byol = byol.to(device)
    classifier = classifier.to(device)

    total_images = 0.0
    total_correct = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            if noisy_inference:
                noise = sigma * torch.randn(imgs.shape).to(device)
                imgs += noise
            imgs = byol.get_representation(imgs)
            imgs = F.normalize(imgs, dim=1)
            outputs = classifier(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_images

    byol_time = time.ctime(byol_time)
    classifier_time = time.ctime(classifier_time)

    if write_to_log:
        if noisy_inference:
            print(f'The classifier accuracy on the noisy test set is: {accuracy}')
            with open(os.path.join('results', 'results/inference.txt'), 'a') as file1:
                file1.write(
                    f'BYOL was created at {byol_time}, Classifier was created at {classifier_time}:\n Noisy test set Accuracy: {accuracy}\n')
        else:
            print(f'The classifier accuracy on the test set is {accuracy}')
            with open(os.path.join('results', 'results/inference.txt'), 'a') as file1:
                file1.write(
                    f'BYOL was created at {byol_time}, Classifier was created at {classifier_time}:\n Test set Accuracy: {accuracy}\n')

    return accuracy


def save_imgs(test_data, byol, classifier, batch_size=8, sigma=0.1):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
    )

    byol = byol.to(device)
    classifier = classifier.to(device)

    (noisy_imgs, imgs), labels = next(iter(test_loader))
    with torch.no_grad():
        noisy_imgs = noisy_imgs.to(device) + sigma * torch.randn(noisy_imgs.shape).to(device)
        imgs = imgs.to(device)
        labels = labels.to(device)

        imgs_representation = byol.get_representation(imgs)
        imgs = F.normalize(imgs, dim=1)
        outputs = classifier(imgs_representation)
        _, predicted = torch.max(outputs.data, 1)

        noisy_imgs_representation = byol.get_representation(noisy_imgs)
        noisy_imgs = F.normalize(noisy_imgs, dim=1)
        noisy_outputs = classifier(noisy_imgs_representation)
        _, noisy_predicted = torch.max(noisy_outputs.data, 1)

    noisy_imgs = (noisy_imgs / 2 + 0.5).clamp(0.0, 1.0)
    imgs = (imgs / 2 + 0.5).clamp(0.0, 1.0)

    labels_to_classes = {0: "airplane",
                         1: "automobile",
                         2: "bird",
                         3: "cat",
                         4: "deer",
                         5: "dog",
                         6: "frog",
                         7: "horse",
                         8: "ship",
                         9: "truck"
                         }
    fig = plt.figure(figsize=(20, 20))
    for i in range(batch_size):
        ax = plt.subplot(4, 4, 2 * i + 1)
        ax.imshow(np.array(imgs[i].cpu()).transpose(1, 2, 0))
        ax.set_title(f'Predicted Label: {labels_to_classes[predicted[i].item()]}')
        ax2 = plt.subplot(4, 4, 2 * i + 2)
        ax2.imshow(np.array(noisy_imgs[i].cpu()).transpose(1, 2, 0))
        ax2.set_title(f'Predicted Label: {labels_to_classes[noisy_predicted[i].item()]}')

    fig.savefig(os.path.join('results', f'images_with_predictions_sigma={sigma}.png'))


def plot_tsne(X, y, N_samples, dim=2, perplexity=30.0, sigma=0.2):
    # X_shape - [2*N_samples, embedding_dim]
    # y_shape - [N_samples]

    labels_to_classes = {0: "airplane",
                         1: "automobile",
                         2: "bird",
                         3: "cat",
                         4: "deer",
                         5: "dog",
                         6: "frog",
                         7: "horse",
                         8: "ship",
                         9: "truck"
                         }
    if dim < 2 or dim > 3:
        print("OH NO :(")
        raise SystemError("2 <= dim <= 3")

    t_sne = TSNE(n_components=dim, perplexity=perplexity)
    X_embedded = t_sne.fit_transform(X)
    orig_embedded = X_embedded[:N_samples, :]
    noisy_embedded = X_embedded[N_samples:, :]

    colors = cm.rainbow(np.linspace(0, 1, 10))
    if dim == 2:
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(1, 1, 1)

        for l in range(10):
            ax.scatter(orig_embedded[y == l, 0], orig_embedded[y == l, 1], color=colors[l], marker='.',
                       label=f'class={labels_to_classes[l]}')
            ax.scatter(noisy_embedded[y == l, 0], noisy_embedded[y == l, 1], color=colors[l], marker='*', label='Noisy')
        ax.grid()
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
        ax.set_title("2D t-SNE of the embedded images")
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        for l in range(10):
            ax.scatter(orig_embedded[y == l, 0], orig_embedded[y == l, 1], orig_embedded[y == l, 2], color=colors[l],
                       marker='.', label=f'class={labels_to_classes[l]}')
            ax.scatter(noisy_embedded[y == l, 0], noisy_embedded[y == l, 1], noisy_embedded[y == l, 2], color=colors[l],
                       marker='*', label='Noisy')
        ax.grid()
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
        ax.set_title("3D t-SNE of the embedded images")

    plt.savefig(os.path.join('results', f'tsne_{dim}D_on_embedding_sigma={sigma}'), bbox_inches="tight")


def create_embedding_and_plot_tsne(test_data, byol, batch_size=2048, sigma=0.1):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
    )

    byol = byol.to(device)

    (noisy_imgs, imgs), labels = next(iter(test_loader))
    with torch.no_grad():
        noisy_imgs = noisy_imgs.to(device) + sigma * torch.randn(noisy_imgs.shape).to(device)
        imgs = imgs.to(device)
        imgs_representation = byol.get_representation(imgs)
        noisy_imgs_representation = byol.get_representation(noisy_imgs)

    represntation = torch.cat((imgs_representation, noisy_imgs_representation), dim=0).cpu()
    plot_tsne(represntation, labels, batch_size, dim=2, sigma=sigma)
    plot_tsne(represntation, labels, batch_size, dim=3, sigma=sigma)
