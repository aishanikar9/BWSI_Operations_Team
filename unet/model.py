import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module): # two convolutions: conv2d 3x3 then ReLU two times
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # down part of unet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature)) # register new layer as submodule 
            in_channels = feature # make num of input channels into the last iterations number of output channels

        # up part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)) # Tranposed Convolution for upsampling: feature*2 because of concatenated matrices
            self.ups.append(DoubleConv(feature*2, feature)) # Double conv for each layer of upsampling.
            # each up layer: transpose conv, then double conv

        # bottom bottelneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1]*2) # double conv: in is 512, out is 1024 to increase feature channels to its max

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) # final layer: done! single convolution with in 64, out 1

    def forward(self, x):
        skip_connections = []

        for down in self.downs: # "for every layer in encoder:"
            x = down(x)         # run x through the layer
            skip_connections.append(x) # then save it to skip_connections to be catted later
            x = self.pool(x)    # pool it (downsample) uses max pooling to shrink in half

        x = self.bottleneck(x)  # once encoding is done, run through bottleneck layer
        skip_connections = skip_connections[::-1] # reverse skip_connections for convenience so we can use it in decoding

        for idx in range(0, len(self.ups), 2): # for each layer in decoder. 8 layers, step 2 at a time since theres 2 steps at each of 4 layers
            x = self.ups[idx](x)# run x through the first step of the up layer
            skip_connection = skip_connections[idx//2] # make local var skip_connection from skip_connections list we made in encoding

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:]) # resize so catting works. [2:] gets height and width of tensor

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip) # run x with skip connection this time through the second step of the layer

        return self.final_conv(x) # do the final layer single conv and return the result. done!
    

def test(): # test to make sure the size of what you input into the unet comes back the same size
    x = torch.randn((3,1,160,160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()