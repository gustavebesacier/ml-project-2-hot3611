import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.variables.variables import FULL_CNN_FEATURES


def double_conv(in_channels, out_channels):
    """
      Creates a block of two convolutional layers, each followed by BatchNorm and ReLU activation.

      @param in_channels: The number of input channels for the first convolutional layer.
      @param out_channels: The number of output channels for both convolutional layers.
      @return: A sequential model containing two convolutional layers with BatchNorm and ReLU activation.
      """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class FullCNN(nn.Module):
    """
    CNN for road segmentation.
    """

    def __init__(self, in_channels=3, out_channels=1, features=FULL_CNN_FEATURES):
        super(FullCNN, self).__init__()

        # Encoder
        self.encoder = nn.ModuleDict()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        prev_channels = in_channels
        for idx, f in enumerate(features):
            self.encoder[f"enc_{idx}"] = double_conv(prev_channels, f)
            prev_channels = f

        # Decoder
        self.decoder = nn.ModuleDict()
        # We skip the last encoder stage because that one goes straight to the decoder
        for idx in range(len(features) - 1, 0, -1):
            self.decoder[f"upconv_{idx}"] = nn.ConvTranspose2d(features[idx], features[idx-1], kernel_size=2, stride=2)
            self.decoder[f"dec_{idx}"] = double_conv(features[idx], features[idx-1])

        # Segmentation head: final 1x1 convolution to get desired output channels
        self.segmentation_head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder forward
        skip_connections = []
        for idx, layer_key in enumerate(self.encoder.keys()):
            x = self.encoder[layer_key](x)
            if idx < len(self.encoder) - 1:
                # store intermediate features for skip connections
                skip_connections.append(x)
                x = self.pool(x)

        # Reverse skip_connections so that the top-level features are at the start of the list
        skip_connections = skip_connections[::-1]

        # Decoder forward
        # Note: The decoder keys are generated in descending order of indices.
        # For each pair (upconv_X, dec_X), we first upsample, then concatenate skip connections, then double_conv.
        for idx, layer_key in enumerate(self.decoder.keys()):
            if "upconv" in layer_key:
                # Perform upsampling
                x = self.decoder[layer_key](x)
                # Concat with skip connection
                skip_x = skip_connections[idx // 2]
                # If dimensions differ slightly due to odd input sizes, we interpolate
                if x.shape != skip_x.shape:
                    x = F.interpolate(x, size=skip_x.shape[2:], mode='bilinear', align_corners=False)
            else:
                # Apply the double conv after concatenation
                x = torch.cat((skip_x, x), dim=1)
                x = self.decoder[layer_key](x)

        # Segmentation head
        x = self.segmentation_head(x)
        return x


if __name__ == "__main__":
    print("FullCNN")

