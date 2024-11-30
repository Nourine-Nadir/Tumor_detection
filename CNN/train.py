import numpy as np
import torch as T
from models import InceptionModel, ConvModule
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import load_images


def train(features,
          labels,
          config,
          model_path,
          load_model=True
          ):
    # Split the data into training and validation sets
    # Typically, 80-20 or 70-30 split is used
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels,
        test_size=0.2,  # 20% for validation
        random_state=42 # for reproducibility
    )

    # Set up device
    DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')

    # Prepare training data
    batch_size = config['batch_size']
    n_epochs = config['epochs']

    # Convert to tensors
    with T.no_grad():
        # Training data
        train_images = T.FloatTensor(X_train.astype(np.float32) / 255.0)
        train_labels = T.tensor(y_train, device=DEVICE, dtype=T.float32)

        # Validation data
        val_images = T.FloatTensor(X_val.astype(np.float32) / 255.0)
        val_labels = T.tensor(y_val, device=DEVICE, dtype=T.float32)

        # Ensure correct tensor dimensions
        if len(train_images.shape) == 3:
            train_images = train_images.unsqueeze(1)
        if len(val_images.shape) == 3:
            val_images = val_images.unsqueeze(1)

        # Move to device
        train_images = train_images.to(DEVICE)
        val_images = val_images.to(DEVICE)

    # Initialize model
    model = InceptionModel(nb_classes=2,lr=config['lr'])
    if load_model :
        try:
            model.load_model(model_path, map_location=DEVICE)
            print('Model loaded !')

        except:
            print('Failed loading model !')

    # Prepare data loaders
    train_dataset = T.utils.data.TensorDataset(train_images, train_labels)
    train_dataloader = T.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataset = T.utils.data.TensorDataset(val_images, val_labels)
    val_dataloader = T.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False  # No need to shuffle validation data
    )

    # Tracking for plotting
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in tqdm(range(n_epochs)):
        # Training phase
        model.train()
        total_train_loss = 0

        for batch_features, batch_labels in train_dataloader:
            batch_features = batch_features.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            # Forward pass
            predictions = model(batch_features)
            loss = model.loss(predictions, batch_labels)

            # Backward pass
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            total_train_loss += loss.item()

        # Validation phase
        model.eval()
        total_val_loss = 0
        all_predictions = []
        all_true_labels = []
        with T.no_grad():
            for batch_features, batch_labels in val_dataloader:
                batch_features = batch_features.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)

                # Forward pass
                predictions = model(batch_features)
                val_loss = model.loss(predictions, batch_labels)
                total_val_loss += val_loss.item()

                all_predictions.append(predictions.cpu().numpy())
                all_true_labels.append(batch_labels.cpu().numpy())

            all_predictions = np.concatenate(all_predictions, axis=0)
            all_true_labels = np.concatenate(all_true_labels, axis=0)
            print(f'all predictions: {all_predictions.shape}')
            print(f'all true predictions: {all_true_labels.shape}')

            accuracy = accuracy_score(
                np.argmax(all_true_labels, axis=-1),
                np.argmax(all_predictions, axis=-1)
            )
        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)

        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Periodic logging
        if (epoch + 1) % 1 == 0:
            model.lr_decay()
            print(f'accuracy {accuracy:.2f}')
            print(f'Epoch {epoch + 1}/{n_epochs}')
            print(f'Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f} , lr {model.get_lr()}')

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_comparison.png')
    plt.show()

    # Save model
    model.save_model(model_path)

    return model, train_losses
