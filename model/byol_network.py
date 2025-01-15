import torchvision.models as models
import torch
from models.mlp_head import MLPHead
from einops.layers.torch import Rearrange
from models.vit import ViT



class ByolNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ByolNet, self).__init__()
        
        if kwargs['pretrained']:
            self.byolnet = models.vision_transformer.vit_b_16(image_size=224, weights='DEFAULT')
            in_channels = self.byolnet.heads.head.in_features
        else:
            self.byolnet =  ViT(image_size = 32,
                                patch_size = 4,
                                num_classes = 10,
                                dim = 512, 
                                depth = 6,
                                heads = 8,
                                mlp_dim = 512,
                                dropout = 0.1,
                                emb_dropout = 0.1
                                
                            )

            in_channels = self.byolnet.mlp_head[-1].in_features
            
        self.pretrained = kwargs['pretrained']

        self.projection = MLPHead(in_channels=in_channels, **kwargs['projection_head'])

    def forward(self, x):
        
        if self.pretrained:
            x = self.byolnet._process_input(x)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.byolnet.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.byolnet.encoder(x)

            # Classifier "token" as used by standard language architectures
            x = x[:, 0]

        else:
            x = self.byolnet(x)

        x = self.projection(x)

        return x


    def get_representation(self, x):
        
        if self.pretrained:
            x = self.byolnet._process_input(x)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.byolnet.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.byolnet.encoder(x)

            # Classifier "token" as used by standard language architectures
            x = x[:, 0]
        else:
            x = self.byolnet(x)
        return x