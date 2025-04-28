import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):

        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self,  input_feature,adjacency):

        batch_size, num_nodes, _ = input_feature.size()
        support = torch.matmul(input_feature, self.weight)
        output = torch.bmm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output


class PatchEmbed(nn.Module):

    def __init__(self, img_size=256, patch_size=16, in_c=3, embed_dim=1024, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # print(x.shape)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes,kernel_size =1,stride=1):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=kernel_size,stride=stride,padding=0),
        )

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class GraphConvBlock(nn.Module):
    def __init__(self, in_channels,dimembed, out_channels, dropout):
        super(GraphConvBlock, self).__init__()
        self.gcn = GraphConvolution(in_channels, dimembed)
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dimembed, out_channels)

    def forward(self, x, adj):
        x = self.gcn(x, adj)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, dim_dim, output_dim, dropout, num_layers):
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(GraphConvBlock(in_channels, dim_dim,output_dim, dropout))

    def forward(self, x, adj):
        for block in self.blocks:
            x = block(x, adj)
        return x


class GKNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64,
                 num_layers_mlp:int = 2,
                 dimembed: int = 2048,
                 d_model:int = 1024,
                 num_decoder_layer:int = 1,
                 dropout:float = 0.0,):

        super(GKNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16 )
        factor = 2 if bilinear else 1
        self.down5 = Down(base_c * 16, base_c * 32//factor)
        self.up1 = Up(base_c * 32, base_c * 16 // factor)
        self.up2 = Up(base_c * 16, base_c * 8 // factor)
        self.up3 = Up(base_c * 8, base_c * 4 // factor)
        self.up4 = Up(base_c * 4, base_c * 2//factor)
        self.up5 = Up(base_c * 2, base_c)
        self.patchembed = PatchEmbed(img_size=8, patch_size=2, in_c = base_c * 32 // factor,embed_dim=d_model, norm_layer=nn.LayerNorm)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model,dimembed,d_model,num_layers_mlp)
        self.out_conv = OutConv(base_c, num_classes,8,8)
        self.conv = OutConv(num_classes, num_classes)
        self.decoder = Decoder(d_model,dimembed,d_model,dropout = dropout,num_layers = num_decoder_layer)
        self.dropout = nn.Dropout(dropout)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):

                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                init.ones_(m.weight)
                init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 0.01)
                init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, adj) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        ident = self.patchembed(x6)
        ident = self.mlp(ident)
        ident = self.decoder(ident, adj)
        ident = self.norm(ident)
        ident = ident.view(ident.shape[0], ident.shape[1], 32, 32)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.out_conv(x)
        x = self.dropout(ident) + x
        x = self.conv(x)
        return x

