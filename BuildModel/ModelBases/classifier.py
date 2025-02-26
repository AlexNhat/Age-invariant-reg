from torch import nn


class classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(classifier, self).__init__()

        self.linear_classifier = nn.Sequential(
                          nn.BatchNorm1d(in_channels, affine=False),
                          nn.Linear(in_channels, num_classes)
                        )
    
    def forward(self, x):
        return self.linear_classifier(x)

