import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import _create_model_training_folder


class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, pretrained=False, use_amp=True, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.pretrained = pretrained
        self.use_amp = use_amp
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        self.sigma = params['sigma']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset, val_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True, shuffle=True)

        freq_for_val = 1
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.1 ,verbose=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()
        best_loss = np.inf
        initial_sigma = 0.2
        for epoch_counter in range(self.max_epochs):
            epoch_loss = 0
            sigma =  ((self.sigma - initial_sigma) / (self.max_epochs - 1)) * epoch_counter + initial_sigma
            self.writer.add_scalar('sigma', sigma, global_step=epoch_counter)
            for (batch_view_1, batch_view_2), _ in tqdm(train_loader, leave=False):

                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)
                               
                noise = (sigma * torch.randn(batch_view_1.shape)).to(self.device)
                batch_view_1 += noise

                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
                    self.writer.add_image('views_1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
                    self.writer.add_image('views_2', grid, global_step=niter)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss = self.update(batch_view_1, batch_view_2)
                epoch_loss += loss.item() 

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    self._update_target_network_parameters()  # update the key encoder
                niter += 1

            if epoch_counter % freq_for_val == 0 or epoch_counter == self.max_epochs - 1:
                val_loss = 0
                self.online_network.eval()
                with torch.no_grad():
                    for (batch_view_1, batch_view_2), _ in val_loader:
                        batch_view_1 = batch_view_1.to(self.device)
                        noise = (sigma * torch.randn(batch_view_1.shape)).to(self.device)
                        batch_view_1 += noise
                        batch_view_2 = batch_view_2.to(self.device)
                        val_loss += self.update(batch_view_1, batch_view_2).item()
                        
                    val_loss = val_loss / len(val_loader)
                self.writer.add_scalar('val_loss', val_loss, global_step=epoch_counter)
                if val_loss < best_loss:
                    self.save_model(os.path.join(model_checkpoints_folder, 'byol_model.pth'))
                    best_loss = val_loss
                    epoch_best_loss = epoch_counter

            scheduler.step()
            
            self.writer.add_scalar('loss', epoch_loss / len(train_loader), global_step=epoch_counter)
            print(f'End of epoch {epoch_counter}, Train Loss is: {(epoch_loss / len(train_loader)):.5f}, Val Loss is: {val_loss}, Best Val Loss is: {best_loss}, Best Loss Epoch is: {epoch_best_loss}')
            

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2)) 

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'predictor_network_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)


class ClassifierTrainer:
    def __init__(self, online_network, target_network, classifier, predictor, optimizer, pretrained, device, **params):
        self.online_network = online_network
        self.classifier = classifier
        self.optimizer = optimizer
        self.device = device
        self.pre_trained = pretrained
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])


    def train(self, train_dataset, val_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=2, verbose=True)

        freq_for_val = 1

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.online_network.to(self.device)
        self.classifier.to(self.device)

        best_accuracy = 0

        self.online_network.eval()
        for epoch_counter in range(self.max_epochs):

            epoch_loss = 0
            self.classifier.train()
            for images, labels in tqdm(train_loader, leave=False):
                
                loss = self.update(images, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(train_loader)
            self.writer.add_scalar('loss', epoch_loss, global_step=epoch_counter)

            
            if epoch_counter % freq_for_val == 0 or epoch_counter == self.max_epochs - 1:
                val_loss = 0
                self.classifier.eval()
                total_images = 0
                total_correct = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        batch_val_loss, outputs = self.update(images, labels, val=True)
                        val_loss += batch_val_loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        predicted = predicted.cpu()
                        total_images += labels.size(0)
                        total_correct += (predicted == labels).sum().item()

                val_accuracy = total_correct / total_images
                val_loss = val_loss / len(val_loader)
                self.writer.add_scalar('val_loss', val_loss, global_step=epoch_counter)
                self.writer.add_scalar('val_accuracy', val_accuracy, global_step=epoch_counter)
                if val_accuracy > best_accuracy:
                    self.save_model(os.path.join(model_checkpoints_folder, 'classifier_model.pth'))
                    best_accuracy = val_accuracy
            
                scheduler.step(val_accuracy)
        
            print(f'End of epoch {epoch_counter}, val_loss={val_loss:.5f}, val_accuracy={val_accuracy:.5f}')
            

    def update(self, images, labels, val=False):
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            # images_embedding = self.target_network(images)
            images_embedding = self.online_network.get_representation(images)
            images_embedding = F.normalize(images_embedding, dim=1)
                
        outputs = self.classifier(images_embedding)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        if val:
            return loss, outputs
        else:
            return loss

    def save_model(self, PATH):
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
        }, PATH)