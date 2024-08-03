import torchvision
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, feat_dim = 2048, output_dim=10):
        super(Model, self).__init__()

        self.feat_dim = feat_dim
        self.output_dim = output_dim

        self.backbone = torchvision.models.resnext101_32x8d(pretrained=True)

        # Fix Initial Layers
        for p in list(self.backbone.children())[:-1]:
            p.requires_grad = False

        # get the structure until the Fully Connected Layer
        modules = list(self.backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        
        # Add new fully connected layers
        self.fc1 = nn.Linear(feat_dim, feat_dim//4) # 2048 -> 512
        self.fc2 = nn.Linear(feat_dim//4, output_dim) # 512 -> 10
       
        self.dropout = nn.Dropout(p=0.6)
        self.activation = nn.ReLU()

    def forward(self, img):
        batch_size = img.shape[0]
        out = self.backbone(img) # get the feature from the pre-trained resnet
        #out = self.dropout(self.fc1(out.view(batch_size, -1)))
        out = self.dropout(self.fc1(out.view(batch_size, -1)))
        out = self.activation(out)
        out = self.fc2(out)
        #out = self.fc3(out) # no dropout at the last layer!

        return out
