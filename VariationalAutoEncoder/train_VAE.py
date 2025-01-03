from models import V_AutoEncoder
import torch as T
import numpy as np


def train_AE(data,
          latent_dim,
          load_model,
          encoder_model_path,
          full_model_path,
          epochs:int=50,
          batch_size:int=64,
          lr:float=1e-3
          ):
    # Convert images to float32 and normalize to [0,1] range
    images = T.FloatTensor(data.astype(np.float32) / 255.0)

    # Add channel dimension if not present
    if len(images.shape) == 3:
        images = images.unsqueeze(1)  # Add channel dimension (B, 1, H, W)

    # Initialize the model, loss function, and optimizer
    model = V_AutoEncoder(input_size= images.shape[-1],latent_dim=latent_dim, lr=lr)
    if load_model:
        try:
            model.load_model(full_model_path,map_location=T.device('cuda'))
            print('Model Loaded !')
        except:
            print('No model found !')

    # Move model to appropriate device
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    print('Model inputs shape : ', images.shape)
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        dataloader = T.utils.data.DataLoader(images,
                                             batch_size=batch_size,
                                             shuffle=True)

        for batch_features in dataloader:
            # Move batch to device
            inputs = batch_features.to(device)
            # Forward pass
            outputs, mu, var = model(inputs)
            reconstruction_loss = model.criterion(outputs, inputs)
            kl_loss = model.kl_loss(mu, var)

            loss = reconstruction_loss + 0.00001 *kl_loss

            # Backward pass and optimize
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            running_loss += loss.item()

        model.lr_decay()
        # Print epoch statistics
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], lr : {model.get_lr():.5f} , Average Loss: {avg_loss:.4f}")

    model.encoder.save_model(encoder_model_path)
    model.save_model(full_model_path)

    return model,model.encoder,inputs.cpu().detach().numpy(), outputs.cpu().detach().numpy()