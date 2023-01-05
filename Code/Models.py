import torch.nn as nn
import torch.nn.init as init
import torch
import torchvision.models as models

def double_conv(in_channels, out_channels,affine):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels,affine=affine),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels,affine=affine),
        nn.ReLU(inplace=True))



def conv_bn_relu(in_channels, out_channels, kernel_size,affine=False):
    layer = []
    layer.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False))
    layer.append(nn.BatchNorm2d(out_channels,affine=affine))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)


def conv_bn_relu_transpose(in_channels, out_channels, kernel_size,affine=False):
    layer = []
    layer.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, bias=False))
    layer.append(nn.BatchNorm2d(out_channels,affine=affine))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)

def conv_bn_relu_dropout(in_channels, out_channels, kernel_size,affine=False):
    layer = []
    layer.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False))
    layer.append(nn.BatchNorm2d(out_channels,affine=affine))
    layer.append(nn.ReLU(inplace=True))
    layer.append(nn.Dropout2d(p=0.5))
    return nn.Sequential(*layer)


def conv_bn_relu_dropout_transpose(in_channels, out_channels, kernel_size,affine=False):
    layer = []
    layer.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, bias=False))
    layer.append(nn.BatchNorm2d(out_channels,affine=affine))
    layer.append(nn.ReLU(inplace=True))
    layer.append(nn.Dropout2d(p=0.5))
    return nn.Sequential(*layer)

class FCRNWrapper(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, sigmoid=False, affine=False):

        super(FCRNWrapper, self).__init__()
        self.ecml = FCRNECML(in_channels,out_channels,kernel_size,sigmoid,affine)
        self.edge_decoder = nn.Sequential(nn.ConvTranspose2d(out_channels * 16, out_channels * 4, 2, stride=2, bias=False),
                                      nn.BatchNorm2d(out_channels * 4, affine=affine),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(out_channels * 4, out_channels * 2, 2, stride=2, bias=False),
                                      nn.BatchNorm2d(out_channels * 2, affine=affine),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(out_channels * 2, out_channels, 2, stride=2, bias=False),
                                      nn.BatchNorm2d(out_channels, affine=affine),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(out_channels, in_channels, 3, padding=1))

    def forward(self,x):
        _,feat = self.ecml(x)
        out = self.edge_decoder(feat)
        return out,0

class FCRN(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, sigmoid=False, affine=False):

        super(FCRN, self).__init__()
        self.add_sigmoid = sigmoid
        """
        # Encoder

        self.conv1 = conv_bn_relu(in_channels, out_channels, kernel_size,affine=affine)
        self.conv2 = conv_bn_relu(out_channels, out_channels * 2, kernel_size,affine=affine)
        self.conv3 = conv_bn_relu(out_channels * 2, out_channels * 4, kernel_size,affine=affine)
        # LatentSpace
        self.conv4 = conv_bn_relu(out_channels * 4, out_channels * 16, kernel_size, affine=affine)

        self.maxpool = nn.MaxPool2d(2, 2)
        # Decoder
        self.conv5 = conv_bn_relu_transpose(out_channels * 16, out_channels * 4, 2, affine=affine)
        self.conv6 = conv_bn_relu_transpose(out_channels * 4, out_channels * 2, 2, affine=affine)
        self.conv7 = conv_bn_relu_transpose(out_channels * 2, out_channels, 2, affine=affine)
        self.conv8 = nn.Conv2d(out_channels, in_channels, 3, padding=1)
        #self.conv_cvs_pre = conv_bn_relu_transpose(out_channels * 2, out_channels, 2, affine=affine)
        #self.conv_cvs = nn.Conv2d(out_channels, in_channels, 3, padding=1)

        """

        self.encoder = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
                                     nn.BatchNorm2d(out_channels, affine=affine),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2, 2),
                                     nn.Conv2d(out_channels, out_channels * 2, kernel_size, padding=1, bias=False),
                                     nn.BatchNorm2d(out_channels * 2, affine=affine),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2, 2),
                                     nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size, padding=1, bias=False),
                                     nn.BatchNorm2d(out_channels * 4, affine=affine),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2, 2),
                                     nn.Conv2d(out_channels * 4, out_channels * 16, kernel_size, padding=1, bias=False),
                                     nn.BatchNorm2d(out_channels * 16, affine=affine),
                                     nn.ReLU(inplace=True)
                                     )

        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(out_channels * 16, out_channels * 4, 2, stride=2, bias=False),
                                      nn.BatchNorm2d(out_channels * 4, affine=affine),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(out_channels * 4, out_channels * 2, 2, stride=2, bias=False),
                                      nn.BatchNorm2d(out_channels * 2, affine=affine),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(out_channels * 2, out_channels, 2, stride=2, bias=False),
                                      nn.BatchNorm2d(out_channels, affine=affine),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(out_channels, in_channels, 3, padding=1))

        self.decoder2 = nn.Sequential(nn.ConvTranspose2d(out_channels * 16, out_channels * 4, 2, stride=2, bias=False),
                                      nn.BatchNorm2d(out_channels * 4, affine=affine),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(out_channels * 4, out_channels * 2, 2, stride=2, bias=False),
                                      nn.BatchNorm2d(out_channels * 2, affine=affine),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(out_channels * 2, out_channels, 2, stride=2, bias=False),
                                      nn.BatchNorm2d(out_channels, affine=affine),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(out_channels, in_channels, 3, padding=1))

        if self.add_sigmoid:
            self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def forward(self, x, out_idx=1):

        x = self.encoder(x)
        feature_dist = x
        if out_idx == 1:
            out = self.decoder1(x)
        else:
            out = self.decoder2(x)
        """
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.maxpool(self.conv3(x))

        x = self.conv4(x)
        feature_dist = x

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        #if out_idx==1:
         #   x = self.conv7(x)
          #  x = self.conv8(x)
        #else:
         #   x = self.conv_cvs_pre(x)
          #  x = self.conv_cvs(x)
        if self.add_sigmoid:
            out = self.sigmoid(x)
        else:
            out = x
        """
        return out, feature_dist

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    init.constant_(m.weight, 0.1)
                    init.constant_(m.bias, 0)


class FCRNECML(nn.Module):
    def __init__(self,in_channels=1, out_channels=32, kernel_size=3,sigmoid=False,affine=False):

        super(FCRNECML, self).__init__()
        self.add_sigmoid = sigmoid
        # Encoder
        self.conv1 = conv_bn_relu(in_channels, out_channels, kernel_size,affine=affine)
        self.conv2 = conv_bn_relu(out_channels, out_channels * 2, kernel_size,affine=affine)
        self.conv3 = conv_bn_relu(out_channels * 2, out_channels * 4, kernel_size,affine=affine)

        self.maxpool = nn.MaxPool2d(2,2)

        # LatentSpace
        self.conv4 = conv_bn_relu(out_channels * 4, out_channels * 16, kernel_size,affine=affine)

        # Decoder
        self.conv5 = conv_bn_relu_transpose(out_channels * 16, out_channels * 4, 2,affine=affine)
        self.conv6 = conv_bn_relu_transpose(out_channels * 4, out_channels * 2, 2,affine=affine)
        self.conv7 = conv_bn_relu_transpose(out_channels * 2, out_channels, 2,affine=affine)
        self.conv8 = nn.Conv2d(out_channels, in_channels, 3, padding=1)
        if self.add_sigmoid:
            self.sigmoid = nn.Sigmoid()

        self._initialize_weights()
    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.maxpool(self.conv3(x))
        x = self.conv4(x)
        feature_dist_lc = x

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        if self.add_sigmoid:
            out = self.sigmoid(x)
        else:
            out = x

        return out,feature_dist_lc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight)
                #init.orthogonal_(m.weight)
                #init.xavier_normal_(m.weight)
                #init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m,nn.ConvTranspose2d):
                init.normal_(m.weight)
                #init.orthogonal_(m.weight)
                #init.xavier_normal_(m.weight)
                #init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    init.constant_(m.weight, 0.1)
                    init.constant_(m.bias, 0)


class FCRNECML_dropout(nn.Module):
    def __init__(self,in_channels=1, out_channels=32, kernel_size=3,sigmoid=False,affine=False):

        super(FCRNECML_dropout, self).__init__()
        self.add_sigmoid = sigmoid
        # Encoder
        self.conv1 = conv_bn_relu(in_channels, out_channels, kernel_size,affine=affine)
        self.conv2 = conv_bn_relu(out_channels, out_channels * 2, kernel_size,affine=affine)
        self.conv3 = conv_bn_relu(out_channels * 2, out_channels * 4, kernel_size,affine=affine)

        self.maxpool = nn.MaxPool2d(2,2)

        # LatentSpace
        self.conv4 = conv_bn_relu(out_channels * 4, out_channels * 16, kernel_size,affine=affine)

        # Decoder
        self.conv5 = conv_bn_relu_dropout_transpose(out_channels * 16, out_channels * 4, 2,affine=affine)
        self.conv6 = conv_bn_relu_transpose(out_channels * 4, out_channels * 2, 2,affine=affine)
        self.conv7 = conv_bn_relu_dropout_transpose(out_channels * 2, out_channels, 2,affine=affine)
        self.conv8 = nn.Conv2d(out_channels, in_channels, 3, padding=1)
        if self.add_sigmoid:
            self.sigmoid = nn.Sigmoid()

        self._initialize_weights()
    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.maxpool(self.conv3(x))
        x = self.conv4(x)
        feature_dist_lc = x

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        if self.add_sigmoid:
            out = self.sigmoid(x)
        else:
            out = x

        return out,feature_dist_lc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight)
                #init.orthogonal_(m.weight)
                #init.xavier_normal_(m.weight)
                #init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m,nn.ConvTranspose2d):
                init.normal_(m.weight)
                #init.orthogonal_(m.weight)
                #init.xavier_normal_(m.weight)
                #init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    init.constant_(m.weight, 0.1)
                    init.constant_(m.bias, 0)
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, affine=False):
        super(Generator, self).__init__()


        self.conv1 = conv_bn_relu(in_channels, out_channels, kernel_size, affine=affine)
        self.conv2 = conv_bn_relu(out_channels, out_channels * 2, kernel_size, affine=affine)
        self.conv3 = conv_bn_relu(out_channels * 2, out_channels * 4, kernel_size, affine=affine)
        self.maxpool = nn.MaxPool2d(2, 2)
        # LatentSpace
        self.conv4 = conv_bn_relu(out_channels * 4, out_channels * 16, kernel_size, affine=affine)

        self._initialize_weights()
    def forward(self,x):
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.maxpool(self.conv3(x))
        x = self.conv4(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    init.constant_(m.weight, 0.1)
                    init.constant_(m.bias, 0)

class Decoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, sigmoid=False, affine=False):
        super(Decoder, self).__init__()
        self.add_sigmoid = sigmoid
        # Decoder
        self.conv5 = conv_bn_relu_transpose(out_channels * 16, out_channels * 4, 2, affine=affine)
        self.conv6 = conv_bn_relu_transpose(out_channels * 4, out_channels * 2, 2, affine=affine)
        self.conv7 = conv_bn_relu_transpose(out_channels * 2, out_channels, 2, affine=affine)
        self.conv8 = nn.Conv2d(out_channels, in_channels, 3, padding=1)
        if self.add_sigmoid:
            self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def forward(self, x):
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        if self.add_sigmoid:
            out = self.sigmoid(x)
        else:
            out = x
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    init.constant_(m.weight, 0.1)
                    init.constant_(m.bias, 0)


class Discriminator(nn.Module):
    def __init__(self, neurons = 500,n_classes=1):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(nn.Linear(524288,neurons),
                                           nn.Dropout(p=0.5,inplace=True),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(neurons,neurons),
                                           nn.Dropout(p=0.5,inplace=True),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(neurons, n_classes),
                                           nn.LogSoftmax(dim=1))

    def forward(self,x):
        x = x.view(x.size(0),-1)
        return self.discriminator(x)



class UNet(nn.Module):

    def __init__(self, n_class,sigmoid=False,affine=False):
        super(UNet,self).__init__()

        self.dconv_down1 = double_conv(1, 32,affine=affine)
        #self.dconv_down1 = double_conv(1, 16, affine=affine)
        #self.dconv_down2 = double_conv(64, 128,affine=affine)
        #self.dconv_down2 = double_conv(16, 32, affine=affine)
        self.dconv_down2 = double_conv(32, 64, affine=affine)
        self.dconv_down3 = double_conv(64, 128,affine=affine)
        #self.dconv_down3 = double_conv(128, 256, affine=affine)
        #self.dconv_down3 = double_conv(128, 256, affine=affine)
        #self.dconv_down3 = double_conv(32, 32, affine=affine)
        self.dconv_down4 = double_conv(128, 512,affine=affine)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.deconv1 = nn.ConvTranspose2d(32,32,kernel_size=2,stride=2)
        #self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dconv_up3 = double_conv(256 + 512, 256,affine=affine)
        self.dconv_up2 = double_conv(64 + 128, 64,affine=affine)
        #self.dconv_up2 = double_conv(128 + 256, 128, affine=affine)
        #self.dconv_up2 = double_conv(32 + 32, 32, affine=affine)
        #self.dconv_up2 = double_conv(128 + 256, 128, affine=affine)
        #self.dconv_up2 = double_conv(64 + 64, 64, affine=affine)
        #self.dconv_up1 = double_conv(64 + 128, 64,affine=affine)
        #self.dconv_up1 = double_conv(128 + 32, 32, affine=affine)
        self.dconv_up1 = double_conv(32 + 64, 32, affine=affine)
        #self.dconv_up1 = double_conv(32 + 16, 16, affine=affine)
        #self.dconv_up1 = double_conv(16 + 16, 16, affine=affine)
        self.conv_last = nn.Conv2d(32, n_class, kernel_size=1)
        self.add_sigmoid = sigmoid
        if self.add_sigmoid:
            self.sigmoid = nn.Sigmoid()
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight)
                #init.orthogonal_(m.weight)
                #init.xavier_normal_(m.weight)
                #init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight)
                #init.orthogonal_(m.weight)
                #init.xavier_normal_(m.weight)
                #init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    init.constant_(m.weight, 0.1)
                    init.constant_(m.bias, 0)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        x = self.dconv_down3(x)

        feature_distill =x
        x = self.upsample(x)
        #x = self.deconv1(x)
        x = torch.cat([x, conv2], dim=1)



        x = self.dconv_up2(x)
        x = self.upsample(x)
        #x = self.deconv2(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        if self.add_sigmoid:
            out = nn.Sigmoid()(x)
        else:
            out = x
        return out,feature_distill



class UNet2(nn.Module):

    def __init__(self, n_class,sigmoid=False,affine=False):
        super(UNet2,self).__init__()

        self.dconv_down1 = double_conv(1, 32,affine=affine)
        #self.dconv_down1 = double_conv(1, 16, affine=affine)
        #self.dconv_down2 = double_conv(64, 128,affine=affine)
        #self.dconv_down2 = double_conv(16, 32, affine=affine)
        self.dconv_down2 = double_conv(32, 64, affine=affine)
        self.dconv_down3 = double_conv(64, 128,affine=affine)
        #self.dconv_down3 = double_conv(128, 256, affine=affine)
        #self.dconv_down3 = double_conv(128, 256, affine=affine)
        #self.dconv_down3 = double_conv(32, 32, affine=affine)
        #self.dconv_down4 = double_conv(128, 512,affine=affine)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.deconv1 = nn.ConvTranspose2d(32,32,kernel_size=2,stride=2)
        #self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        #self.dconv_up3 = double_conv(256 + 512, 256,affine=affine)
        self.dconv_up2 = double_conv(64 + 128, 64,affine=affine)
        #self.dconv_up2 = double_conv(128 + 256, 128, affine=affine)
        #self.dconv_up2 = double_conv(32 + 32, 32, affine=affine)
        #self.dconv_up2 = double_conv(128 + 256, 128, affine=affine)
        #self.dconv_up2 = double_conv(64 + 64, 64, affine=affine)
        #self.dconv_up1 = double_conv(64 + 128, 64,affine=affine)
        #self.dconv_up1 = double_conv(128 + 32, 32, affine=affine)
        self.dconv_up1 = double_conv(32 + 64, 32, affine=affine)
        #self.dconv_up1 = double_conv(32 + 16, 16, affine=affine)
        #self.dconv_up1 = double_conv(16 + 16, 16, affine=affine)
        self.conv_last = nn.Conv2d(32, n_class, kernel_size=1)
        self.add_sigmoid = sigmoid
        if self.add_sigmoid:
            self.sigmoid = nn.Sigmoid()
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight)
                # init.orthogonal_(m.weight)
                #init.xavier_normal_(m.weight)
                #init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight)
                # init.orthogonal_(m.weight)
                #init.xavier_normal_(m.weight)
                #init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    init.constant_(m.weight, 0.1)
                    init.constant_(m.bias, 0)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        x = self.dconv_down3(x)

        feature_distill =x
        x = self.upsample(x)
        #x = self.deconv1(x)
        x = torch.cat([x, conv2], dim=1)



        x = self.dconv_up2(x)
        x = self.upsample(x)
        #x = self.deconv2(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        if self.add_sigmoid:
            out = nn.Sigmoid()(x)
        else:
            out = x
        return out,feature_distill



class AlexNet(nn.Module):
    def __init__(self, affine=False,num_classes=7):
        super(AlexNet,self).__init__()
        self.affine=affine
        self.alexnet = models.alexnet(pretrained=True)
        #self.alexnet.classifier[6] = nn.Linear(4096,num_classes)
        self.setBatchNormAffine()

    def forward(self, x):
        #return self.alexnet(x)
        x = self.alexnet.features(x)
        x = self.decoder(x)
        return x

    def setBatchNormAffine(self):
        for m in self.alexnet.features.modules():
            if isinstance(m,nn.Sequential):
                for child in m.modules():
                    if isinstance(child,nn.BatchNorm2d):
                            child.affine = self.affine