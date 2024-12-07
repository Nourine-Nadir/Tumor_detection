import torch as T
import numpy as np
from sklearn.model_selection import train_test_split
from vit import ViT
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def train(
        images,
        labels,
        config,
        model_path,
        load_model
):
    DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
    with T.no_grad():  # Add this to prevent gradient computation for encoder
        images = T.FloatTensor(np.array(images).astype(np.float32) / 255.0)
    if len(images.shape) == 3 :
        images = T.unsqueeze(images, dim=1)
    print(images.shape)
    model = ViT(
        image_size = images.shape[-1],
        patch_size = 8,
        num_classes = 2,
        dim = 64,
        depth = 6,
        heads = 4,
        mlp_dim = 128,
        dropout = 0.2,
        emb_dropout = 0.2,
        channels = 1,
    ).to(DEVICE)


    if load_model:
        model.load_model(model_path+ ' best_model')
    # labels = np.argmax(labels, axis=1)
    labels = T.tensor(
        np.argmax(np.array(labels),axis=-1)
    )
    all_labels= labels
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
                                           shuffle=True)
    best_loss = 9999
    train_losses = []
    val_losses = []
    optimizer = T.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = T.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: max(0.995 ** epoch, 1e-1)
    )
    criterion = T.nn.CrossEntropyLoss()
    for epoch in tqdm(range(config['epochs'])):
        epoch_losses = []
        total_train_loss = 0

        model.train()
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            total_train_loss += loss.item()



        with T.no_grad():
            total_val_loss = 0
            for step, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                epoch_losses.append(loss.item())

                _, predicted = T.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                accuracy = correct / len(labels)
                print(f'Batch Accuracy: {accuracy:.4f}')


            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            if avg_val_loss < best_loss:
                model.save_model(model_path+ ' best_model')
        if epoch % 5 == 0:
            print(f'Train Loss: {avg_train_loss:.6f},'
                  f' Validation Loss: {avg_val_loss:.6f},'
                  )


    model.save_model(model_path+ ' last_model')
    return model