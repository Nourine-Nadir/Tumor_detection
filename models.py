import torch.nn.functional as F
import torch.nn as nn
import torch as T
import numpy  as np
import os



def weights_init_(m):
    if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform_(m.weight, gain=1)
        T.nn.init.constant_(m.bias, 0)

class Naive_net(nn.Module):
    def __init__(self,
                 lr,
                 input_shape,
                 fc1_dims,
                 fc2_dims,
                 fc3_dims,
                 fc4_dims,
                 n_output):
        super(Naive_net, self).__init__()
        self.fc1 = nn.Linear(input_shape, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc2_dims)
        self.fc4 = nn.Linear(fc2_dims, fc3_dims)
        self.fc5 = nn.Linear(fc3_dims, fc4_dims)
        self.fc6 = nn.Linear(fc4_dims, n_output)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        self.apply(weights_init_)

        self.loss= nn.CrossEntropyLoss()
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(0.99 ** epoch, 1e-2 )
        )

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        print(f'device used: {self.device}')
        self.to(self.device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = T.FloatTensor(x).to(self.device)
        elif isinstance(x, T.Tensor):
            x = x.to(self.device)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.dropout2(x)
        x = F.relu(self.fc5(x))
        x = F.softmax(self.fc6(x), dim=-1)

        return x

    def lr_decay(self):
        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]

    def save_model(self, PATH):
        os.makedirs(os.path.dirname(PATH), exist_ok=True)

        T.save(self.state_dict(), PATH)

    def load_model(self, PATH, map_location=None):
        if map_location is None:
            map_location = self.device  # Use the model's current device
        self.load_state_dict(T.load(PATH, map_location=map_location, weights_only=True))

class Encoder(nn.Module):
    def __init__(self, input_size=128, latent_dim=64):
        super(Encoder, self).__init__()
        self.input_size = input_size
        # First convolution: 128x128 -> 64x64
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Second convolution: 64x64 -> 32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Third convolution: 32x32 -> 16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Calculate the flattened size
        self.flatten_size = 64 * (input_size // (2 ** 3)) * (input_size // (2 ** 3))

        # Fully connected layer
        self.fc = nn.Linear(self.flatten_size, latent_dim)
        self.apply(weights_init_)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 128x128 -> 64x64
        x = F.relu(self.bn2(self.conv2(x)))  # 64x64 -> 32x32
        x = F.relu(self.bn3(self.conv3(x)))  # 32x32 -> 16x16
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

    def save_model(self, PATH):
        os.makedirs(os.path.dirname(PATH), exist_ok=True)

        T.save(self.state_dict(), PATH)

    def load_model(self, PATH, map_location=None):
        if map_location is None:
            map_location = self.device  # Use the model's current device
        self.load_state_dict(T.load(PATH, map_location=map_location, weights_only=True))


class Decoder(nn.Module):
    def __init__(self, input_size=64, latent_dim=64):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.flatten_size = 64 * (input_size // (2 ** 3)) * (input_size // (2 ** 3))

        # Fully connected layer
        self.fc = nn.Linear(latent_dim, self.flatten_size)

        # First deconvolution: 16x16 -> 32x32
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second deconvolution: 32x32 -> 64x64
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # Third deconvolution: 64x64 -> 128x128
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

        self.apply(weights_init_)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # Reshape from the latent space
        x = self.fc(x)
        x = x.view(x.size(0), 64, self.input_size // (2 ** 3), self.input_size // (2 ** 3))
        # Upsample
        x = F.relu(self.bn1(self.deconv1(x)))  # 16x16 -> 32x32
        x = F.relu(self.bn2(self.deconv2(x)))  # 32x32 -> 64x64
        x = T.sigmoid(self.deconv3(x))  # 64x64 -> 128x128
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_size=128, latent_dim=64, lr=1e-4):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_size, latent_dim)
        self.decoder = Decoder(input_size, latent_dim)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(0.995 ** epoch,1e-2 )
        )
    def forward(self, x):
        if isinstance(x, T.Tensor):
            x = x.to(self.device)
        if isinstance(x, np.ndarray):
            x = T.FloatTensor(x).to(self.device)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def lr_decay(self):
        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]

    def save_model(self, PATH):
        os.makedirs(os.path.dirname(PATH), exist_ok=True)

        T.save(self.state_dict(), PATH)

    def load_model(self, PATH, map_location=None):
        if map_location is None:
            map_location = self.device  # Use the model's current device
        self.load_state_dict(T.load(PATH, map_location=map_location, weights_only=True))


class V_Encoder(nn.Module):
    def __init__(self, input_size=128, latent_dim=64):
        super(V_Encoder, self).__init__()
        self.input_size = input_size
        # First convolution: 128x128 -> 64x64
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Second convolution: 64x64 -> 32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Third convolution: 32x32 -> 16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Calculate the flattened size
        self.flatten_size = 64 * (input_size // (2 ** 3)) * (input_size // (2 ** 3))

        # Fully connected layer
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)
        self.apply(weights_init_)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def reparameterize(self, mu, log_var):
        std = T.exp(0.5*log_var)
        eps = T.randn_like(std)
        z= mu + eps * std

        return z

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 128x128 -> 64x64
        x = F.relu(self.bn2(self.conv2(x)))  # 64x64 -> 32x32
        x = F.relu(self.bn3(self.conv3(x)))  # 32x32 -> 16x16
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        z = self.reparameterize(mu, var)
        return z, mu, var

    def save_model(self, PATH):
        os.makedirs(os.path.dirname(PATH), exist_ok=True)

        T.save(self.state_dict(), PATH)

    def load_model(self, PATH, map_location=None):
        if map_location is None:
            map_location = self.device  # Use the model's current device
        self.load_state_dict(T.load(PATH, map_location=map_location, weights_only=True))


class V_Decoder(nn.Module):
    def __init__(self, input_size=64, latent_dim=64):
        super(V_Decoder, self).__init__()
        self.input_size = input_size
        self.flatten_size = 64 * (input_size // (2 ** 3)) * (input_size // (2 ** 3))

        # Fully connected layer
        self.fc = nn.Linear(latent_dim, self.flatten_size)

        # First deconvolution: 16x16 -> 32x32
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second deconvolution: 32x32 -> 64x64
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        # Third deconvolution: 64x64 -> 128x128
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

        self.apply(weights_init_)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # Reshape from the latent space
        x = self.fc(x)
        x = x.view(x.size(0), 64, self.input_size // (2 ** 3), self.input_size // (2 ** 3))
        # Upsample
        x = F.relu(self.bn1(self.deconv1(x)))  # 16x16 -> 32x32
        x = F.relu(self.bn2(self.deconv2(x)))  # 32x32 -> 64x64
        x = T.sigmoid(self.deconv3(x))  # 64x64 -> 128x128
        return x


class V_AutoEncoder(nn.Module):
    def __init__(self, input_size=128, latent_dim=64, lr=1e-4):
        super(V_AutoEncoder, self).__init__()
        self.encoder = V_Encoder(input_size, latent_dim)
        self.decoder = V_Decoder(input_size, latent_dim)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(0.995 ** epoch,1e-2 )
        )



    def kl_loss(self, mu, log_var):
        kl = -0.5 * T.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return kl.mean()

    def forward(self, x):
        if isinstance(x, T.Tensor):
            x = x.to(self.device)
        if isinstance(x, np.ndarray):
            x = T.FloatTensor(x).to(self.device)

        z, mu, var = self.encoder(x)

        decoded = self.decoder(z)

        return decoded, mu, var

    def lr_decay(self):
        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]

    def save_model(self, PATH):
        os.makedirs(os.path.dirname(PATH), exist_ok=True)

        T.save(self.state_dict(), PATH)

    def load_model(self, PATH, map_location=None):
        if map_location is None:
            map_location = self.device  # Use the model's current device
        self.load_state_dict(T.load(PATH, map_location=map_location, weights_only=True))


class ConvModule(nn.Module):
    def __init__(self, num_filters, kernel_size, stride, padding='same'):
        super(ConvModule, self).__init__()
        self.conv = nn.LazyConv2d(out_channels=num_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class InceptionModule(nn.Module):
    def __init__(self, num_filters1, num_filters2):
        super(InceptionModule, self).__init__()
        self.conv1 = ConvModule(num_filters1, 3, 1)
        self.conv2 = ConvModule(num_filters2, 3, 1)
        self.cat = T.cat

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.cat([x1, x2], dim=1)  # Concatenate along the channel axis
        return x


class DownsampleModule(nn.Module):
    def __init__(self, num_filters):
        super(DownsampleModule, self).__init__()
        self.conv = ConvModule(num_filters, 3, 2, padding=1)
        self.mp = nn.MaxPool2d(2, 2)
        self.cat = T.cat

    def forward(self, x):
        conv_x = self.conv(x)
        pool_x = self.mp(x)
        x = self.cat([conv_x, pool_x], dim=1)  # Concatenate along the channel axis
        return x


class InceptionModel(nn.Module):
    def __init__(self, nb_classes=10, lr=1e-4):
        super(InceptionModel, self).__init__()

        # First Block
        self.conv_1 = ConvModule(96, 3, 1)

        # Second Block
        self.incep_2_1 = InceptionModule(32, 32)
        self.incep_2_2 = InceptionModule(32, 48)
        self.downsample_2 = DownsampleModule(80)

        # Third Block
        self.incep_3_1 = InceptionModule(112, 48)
        self.incep_3_2 = InceptionModule(96, 64)
        self.incep_3_3 = InceptionModule(80, 80)
        self.incep_3_4 = InceptionModule(48, 96)
        self.downsample_3 = DownsampleModule(96)

        # Fourth Block
        self.incep_4_1 = InceptionModule(176, 160)
        self.incep_4_2 = InceptionModule(176, 160)
        self.avg_pool_4 = nn.AvgPool2d(kernel_size=7)

        self.flat_4 = nn.Flatten()
        self.dense_4 = nn.LazyLinear(nb_classes)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.loss = nn.BCELoss()
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(0.95 ** epoch, 1e-1)
        )

    def forward(self, _inputs):
        # Forward pass
        x = self.conv_1(_inputs)

        x = self.incep_2_1(x)
        x = self.incep_2_2(x)
        x = self.downsample_2(x)

        x = self.incep_3_1(x)
        x = self.incep_3_2(x)
        x = self.incep_3_3(x)
        x = self.incep_3_4(x)
        x = self.downsample_3(x)

        x = self.incep_4_1(x)
        x = self.incep_4_2(x)
        x = self.avg_pool_4(x)

        x = self.flat_4(x)
        return F.softmax(self.dense_4(x), dim=-1)

    def lr_decay(self):
        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]

    def save_model(self, PATH):
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        T.save(self.state_dict(), PATH)

    def load_model(self, PATH, map_location=None):
        if map_location is None:
            map_location = self.device  # Use the model's current device
        self.load_state_dict(T.load(PATH, map_location=map_location, weights_only=True))
