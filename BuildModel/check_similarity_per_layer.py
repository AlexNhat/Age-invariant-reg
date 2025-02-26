import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.byol_network import ByolNet


def regression_loss(x, y):
    x = torch.reshape(x, (x.shape[0], -1))
    y = torch.reshape(y, (y.shape[0], -1))
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return (x * y).sum(dim=-1).mean()

def check_similarity_per_layer(byol_model: ByolNet, test_dataset, batch_size=128, sigma=0.1):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
    )
    
    modulelist = ['identity'] + [byol_model.byolnet.conv_proj] + list(byol_model.byolnet.encoder.layers) + [byol_model.projection]
    layer_cosine_loss = {str(layer).partition('(')[0]+f"_{i}": 0 for i,layer in enumerate(modulelist)}
    # layer_l2_loss = {str(layer).partition('(')[0]+f"_{i}": 0 for i,layer in enumerate(modulelist)}
    for (batch_view_1, batch_view_2), _ in tqdm(test_loader, leave=False):
        noise = sigma * torch.randn(batch_view_1.shape).to(device)
        noisy_output = batch_view_1.to(device) + noise
        output = batch_view_2.to(device)
        with torch.no_grad():
            for i,layer in enumerate(modulelist):
                layer_name = str(layer).partition('(')[0]+f"_{i}"

                if i == 1:
                    noisy_output = byol_model.byolnet._process_input(noisy_output)
                    output= byol_model.byolnet._process_input(output)
                    n = output.shape[0]
                    # Expand the class token to the full batch
                    batch_class_token = byol_model.byolnet.class_token.expand(n, -1, -1)
                    output = torch.cat([batch_class_token, output], dim=1)
                    noisy_output = torch.cat([batch_class_token, noisy_output], dim=1)

                elif i == len(modulelist) - 2: #The last layer before the projection module
                    noisy_output = layer(noisy_output)
                    output= layer(output)
                    noisy_output = noisy_output[:,0]
                    output = output[:,0]

                elif i != 0:
                    noisy_output = layer(noisy_output)
                    output= layer(output)
                layer_cosine_loss[layer_name] += regression_loss(noisy_output,output).item()/len(test_loader)
                # layer_l2_loss[layer_name] += torch.mean((noisy_output-output)**2).item()/len(test_loader)
                

    fig = plt.figure(figsize=(30,10))
    plt.plot(list(layer_cosine_loss.keys()),list(layer_cosine_loss.values()), label='Cosine similarity')
    # plt.plot(list(layer_l2_loss.keys()),list(layer_l2_loss.values()), label='MSE')
    # plt.legend()
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity between the original image and the noisy imgae (higher is better)')
    plt.savefig('cosine_similiarity_per_layer.png')
    
    return None