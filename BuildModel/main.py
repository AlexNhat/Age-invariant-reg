import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torchvision.transforms as transforms
import wandb
import yaml
from torchvision import datasets

from check_similarity_per_layer import check_similarity_per_layer
from classifier_inference import classifier_inference, load_weigths, save_imgs, create_embedding_and_plot_tsne
from data.multi_view_data_injector import MultiViewDataInjector
from define_model import define_model
from optuna_exp import objective

print(torch.__version__)

wandb.init(project="vit_byol", sync_tensorboard=True, entity='vit_byol')


def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    sigma = config['trainer']['sigma']
    print(f"Running in mode {config['mode']}")
    if config['network']['pretrained']:
        tf = transforms.Compose([transforms.Resize(size=(224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    else:
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

    if config['mode'] == 'optuna':
        # now we can run the experiment
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(study_name="ViT-BYOL-CIFAR10", direction="minimize", sampler=sampler)

        study.optimize(objective, n_trials=30)

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    elif config['mode'] == 'train_byol':

        trainer = define_model()
        dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=True, download=True,
                                   transform=MultiViewDataInjector([tf, tf]))
        val_size = int(len(dataset) * 0.2)
        train_size = len(dataset) - val_size
        train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, val_size])

        trainer.train(train_data, valid_data)

    elif config['mode'] == 'train_classifier':
        dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=True, download=True, transform=tf)

        val_size = int(len(dataset) * 0.2)
        train_size = len(dataset) - val_size
        train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, val_size])

        classifier_trainer = define_model(train_byol=False)
        classifier_trainer.train(train_dataset=train_data, val_dataset=valid_data)

    elif config['mode'] == 'test_classifier':

        test_dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=False, download=True, transform=tf)
        byol_model, classifier_model, byol_time, classifier_time = load_weigths(config=config)
        classifier_inference(test_data=test_dataset, byol=byol_model, classifier=classifier_model, byol_time=byol_time,
                             classifier_time=classifier_time)
        classifier_inference(test_data=test_dataset, byol=byol_model, classifier=classifier_model, byol_time=byol_time,
                             classifier_time=classifier_time, noisy_inference=True, sigma=sigma)

        test_dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=False, download=True,
                                        transform=MultiViewDataInjector([tf, tf]))
        save_imgs(test_dataset, byol_model, classifier_model, sigma=sigma)
        create_embedding_and_plot_tsne(test_dataset, byol_model, sigma=sigma)

    elif config['mode'] == 'check_similarity_per_layer':
        byol_model, _, _, _ = load_weigths(config=config)
        test_dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=False, download=True,
                                        transform=MultiViewDataInjector([tf, tf]))

        check_similarity_per_layer(byol_model, test_dataset, sigma=sigma)

    elif config['mode'] == 'inference_for_different_sigmas':
        test_dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=False, download=True, transform=tf)
        byol_model, classifier_model, byol_time, classifier_time = load_weigths(config=config)
        sigmas = np.arange(0, 1.55, 0.05)
        accuracy_list = []
        for sigma in sigmas:
            accuracy = classifier_inference(test_data=test_dataset, byol=byol_model, classifier=classifier_model,
                                            byol_time=byol_time, classifier_time=classifier_time, noisy_inference=True,
                                            sigma=sigma, write_to_log=False)
            accuracy_list.append(accuracy)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(sigmas, accuracy_list)
        ax.set_title('Accuracy for different sigmas, BYOL trained on sigma=0.2')
        ax.set_xlabel('sigma')
        ax.set_ylabel('Accuracy')
        fig.savefig('accuracy_vs_sigma.png')
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
