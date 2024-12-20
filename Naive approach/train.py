from models import Naive_net
import numpy as np
import torch as T
def train(features,
          labels,
          lr:float,
          fc1_dims:int,
          fc2_dims:int,
          fc3_dims:int,
          fc4_dims:int,
          n_epochs : int = 100,
          batch_size : int = 32,
          ):
    T.cuda.empty_cache()
    T.backends.cudnn.benchmark = True
    net = Naive_net(lr=lr,
                    input_shape=features.shape[-1],
                    fc1_dims=fc1_dims,
                    fc2_dims=fc2_dims,
                    fc3_dims=fc3_dims,
                    fc4_dims=fc4_dims,
                    n_output=labels.shape[-1])

    features = T.tensor(features, device=net.device, dtype=T.float32)  # Shape: (968, 1)
    labels = T.tensor(labels, device=net.device, dtype=T.float32)
    print('Features shape : ', features.shape,)
    print('labels shape : ', labels.shape, '\n')

    losses = []
    for epoch in range(n_epochs):
            total_loss = 0
            # Create data loader

            dataset = T.utils.data.TensorDataset(features, labels)
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
    net.save_model("models/NaiveNet")

    return net, losses