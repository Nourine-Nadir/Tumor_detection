from models import Naive_net
import numpy as np
import torch as T
from models import Encoder
import matplotlib.pyplot as plt
def train(features,
          labels,
          config,
          encoder_model_path,
          net_model_path,
          load_model
          ):
    n_epochs = config['epochs']
    batch_size = config['batch_size']
    net = Naive_net(lr=config['lr'],
                    input_shape=config['latent_dim'],
                    fc1_dims=config['fc1_dims'],
                    fc2_dims=config['fc2_dims'],
                    fc3_dims=config['fc3_dims'],
                    fc4_dims=config['fc4_dims'],
                    n_output=labels.shape[-1])
    if load_model:
        net.load_model(net_model_path)

    labels = T.tensor(labels, device=net.device, dtype=T.float32)

    encoder = Encoder(latent_dim=config['latent_dim'])
    encoder.load_model(encoder_model_path, map_location=T.device('cuda'))

    with T.no_grad():  # Add this to prevent gradient computation for encoder
        images = T.FloatTensor(features.astype(np.float32) / 255.0)
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        images = images.to(encoder.device)

        # Get encoded features
        encoded_features = encoder(images)
        # Detach to prevent backward through encoder
        encoded_features = encoded_features.detach()

    print('Encoded Features shape : ', encoded_features.shape, '\n')

    losses = []
    for epoch in range(n_epochs):
            net.train()
            total_loss = 0
            # Create data loader

            dataset = T.utils.data.TensorDataset(encoded_features, labels)
            dataloader = T.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True)

            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(net.device)
                batch_labels = batch_labels.to(net.device)

                # Forward pass
                predictions = net(batch_features)

                loss = net.loss(predictions, batch_labels)
                # Backward pass
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()

                total_loss += loss.item()

            losses.append(total_loss)
            if (epoch + 1) % 10 == 0:

                print(f'Epoch {epoch + 1}/{n_epochs}, lr : {net.get_lr():.5f} Loss: {loss:.4f}')
                net.lr_decay()

    plt.plot(losses)
    plt.savefig('loss.png')
    plt.show()

    net.save_model(net_model_path)

    return net, losses