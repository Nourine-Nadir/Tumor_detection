import torch as T
import numpy as np
from sklearn.model_selection import train_test_split
from ViT_model import ViT
from tqdm import tqdm

def train(
        images,
        labels,
        config,
        model_path,
        load_model
):
    with T.no_grad():  # Add this to prevent gradient computation for encoder
        images = T.FloatTensor(np.array(images).astype(np.float32) / 255.0)
    if len(images.shape) == 3 :
        images = T.unsqueeze(images, dim=1)

    model = ViT(
        lr=config['lr'],
        in_channels=images.shape[1],
        img_size=images.shape[-1],
        patch_size=config['patch_size'],
        emb_dim=config['emb_dim'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        mlp_dim=config['mlp_dim'],
        dropout=config['dropout'],
        out_dim=len(np.unique(labels)),

    )
    if load_model:
        model.load_model(model_path)
    # labels = np.argmax(labels, axis=1)
    labels = T.tensor(np.array(labels), dtype=T.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=0.2,  # 20% for validation
        random_state=42  # for reproducibility
    )
    # y_train = T.tensor(np.array(y_train), dtype=T.float32)
    # y_val = T.tensor(np.array(y_val), dtype=T.float32)

    train_dataset = T.utils.data.TensorDataset(X_train, y_train)
    train_loader = T.utils.data.DataLoader(train_dataset,
                                           batch_size=config['batch_size'],
                                           shuffle=True)
    val_dataset = T.utils.data.TensorDataset(X_val, y_val)
    val_loader = T.utils.data.DataLoader(val_dataset,
                                           batch_size=config['batch_size'],
                                           shuffle=False)
    best_loss = 9999
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(config['epochs'])):
        epoch_losses = []
        total_train_loss = 0

        model.train()
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            model.optimizer.zero_grad()

            outputs = model(inputs)
            loss = model.criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()
            epoch_losses.append(loss.item())
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        for step, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            loss = model.criterion(outputs, labels)
            total_val_loss += loss.item()
            epoch_losses.append(loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_loss:
            model.save_model(model_path+ ' best_model')
        if epoch % 1 == 0:
            model.lr_decay()
            print(f'Train Loss: {avg_train_loss:.6f},'
                  f' Validation Loss: {avg_val_loss:.6f},'
                  f'lr {model.get_lr():.5f}' )

    model.save_model(model_path+ ' last_model')
    return model