import optuna
import torch
import torchvision
import torchvision.transforms as transforms

from data.multi_view_data_injector import MultiViewDataInjector
from define_model import define_model

def get_cifar10(batch_size, tf1, tf2):
    dataset= torchvision.datasets.CIFAR10('/tmp/ramdisk/data/', train=True, download=True, transform=MultiViewDataInjector([tf1, tf2]))
    
    val_size = int(len(dataset)*0.2)
    train_size = len(dataset) - val_size
    train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, valid_loader

def objective(trial,n_train_batches=30,n_valid_batches=10):
    
    trainer = define_model()
    trainer.max_epochs = 50
    # Generate the model.
    model = trainer.online_network.to(trainer.device)
    model_predictor = trainer.predictor.to(trainer.device)

    # Generate the optimizers.
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)  # log=True, will use log scale to interplolate between lr
    trainer.optimizer = torch.optim.RAdam(list(model.parameters()) + list(model_predictor.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # Get the CIFAR10 dataset.
    tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Resize(size=(224, 224)),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    train_loader, valid_loader = get_cifar10(batch_size=trainer.batch_size, tf1=tf, tf2=tf)

    initial_sigma = 0.1 
    sigma_max = trainer.sigma 

    # Training of the model.
    for epoch in range(trainer.max_epochs):
        model.train()  
        model_predictor.train()
        n_iter = 0
        sigma =  ((sigma_max - initial_sigma) / (trainer.max_epochs - 1)) * epoch + initial_sigma
        for (batch_view_1, batch_view_2), _ in train_loader:
            # Limiting training data for faster epochs.
            if n_iter  >= n_train_batches:
                break

            batch_view_1 = batch_view_1.to(trainer.device)
            batch_view_2 = batch_view_2.to(trainer.device)
            
            noise = (sigma * torch.randn(batch_view_1.shape)).to(trainer.device)
            batch_view_1 += noise

            loss = trainer.update(batch_view_1, batch_view_2)

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            trainer._update_target_network_parameters()  # update the key encoder
            n_iter += 1
        
        # Validation of the model.
        model.eval()
        model_predictor.eval()
        val_loss = 0
        val_niter = 0
        with torch.no_grad():
            for (batch_view_1, batch_view_2), _ in valid_loader:
                # Limiting validation data.
                if val_niter  >= n_valid_batches:
                    break

                batch_view_1 = batch_view_1.to(trainer.device)
                noise = (sigma * torch.randn(batch_view_1.shape)).to(trainer.device)
                batch_view_1 += noise
                batch_view_2 = batch_view_2.to(trainer.device)
                val_loss += (trainer.update(batch_view_1, batch_view_2)).item()
                val_niter += 1
        val_loss = val_loss / val_niter

        # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
        trial.report(val_loss, epoch)  

        # then, Optuna can decide if the trial should be pruned
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

