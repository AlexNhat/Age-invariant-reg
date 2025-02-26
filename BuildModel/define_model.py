import os

import torch
import yaml

from models.byol_network import ByolNet
from models.classifier import classifier
from models.mlp_head import MLPHead
from trainer import BYOLTrainer, ClassifierTrainer


def define_model(train_byol=True):
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")
   

    # online network
    online_network = ByolNet(**config['network']).to(device)

    # predictor network
    predictor = MLPHead(in_channels=online_network.projection.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = ByolNet(**config['network']).to(device)

    optimizer = torch.optim.Adam(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])
    if train_byol:
        trainer = BYOLTrainer(online_network=online_network,
                            target_network=target_network,
                            optimizer=optimizer,
                            predictor=predictor,
                            device=device,
                            pretrained=config['network']['pretrained'],
                            use_amp=config['network']['use_amp'],
                            **config['trainer'])
        
        return trainer

    else:
        byol_folder = config['network']['checkpoints']
        if byol_folder:
            try:
                load_params = torch.load(os.path.join(os.path.join(byol_folder, 'byol_model.pth')),
                                            map_location=torch.device(torch.device(device)))
                online_network.load_state_dict(load_params['online_network_state_dict'])
                target_network.load_state_dict(load_params['target_network_state_dict'])
                predictor.load_state_dict(load_params['predictor_network_state_dict'])

            except FileNotFoundError:
                print("Pre-trained weights not found.")
                raise FileNotFoundError

        num_classes = 10
        if config['network']['pretrained']:
            in_channels = online_network.byolnet.heads.head.in_features
        else:
            in_channels = online_network.byolnet.mlp_head[-1].in_features
            
        classifier_model = classifier(in_channels=in_channels,num_classes=num_classes)
        classifier_optimizer = torch.optim.Adam(classifier_model.parameters(),
                                **config['optimizer']['classifier_params'])
        
        trainer = ClassifierTrainer(online_network=online_network,
                                    target_network=target_network,
                                    optimizer=classifier_optimizer,
                                    classifier=classifier_model,
                                    predictor=predictor,
                                    pretrained=config['network']['pretrained'],
                                    device=device,
                                    **config['classifier_trainer'])
        return trainer