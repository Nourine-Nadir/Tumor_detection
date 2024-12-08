import torch as T
import numpy as np
from sklearn.model_selection import train_test_split
from ViT.ViT_model import ViT
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
def test(
        model,
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

    DEVICE = model.device
    labels = T.tensor(np.array(labels), device=DEVICE, dtype=T.float32)
    print(type(images))
    test_dataset = T.utils.data.TensorDataset(images, labels)
    test_loader = T.utils.data.DataLoader(test_dataset,
                                           batch_size=config['batch_size'],
                                           shuffle=False)
    all_preds = []
    all_labels = []
    model.eval()  # Set model to evaluation mode
    with T.no_grad():
        for inputs, batch_labels in test_loader:
            inputs = inputs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            predictions = model(inputs)
            pred_labels = predictions.cpu().numpy()

            all_preds.extend(pred_labels)
            all_labels.extend(batch_labels.cpu().numpy())

    labels_1D = T.argmax(labels, dim=1).cpu().numpy()
    all_preds_1D = np.argmax(all_preds,axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(labels_1D, all_preds_1D)

    # Calculate loss
    loss = model.criterion(T.tensor(np.array(all_labels), device=DEVICE),
                           T.tensor(np.array(all_preds), device=DEVICE)).item()

    # Calculate f1score
    f1score = f1_score(labels_1D, all_preds_1D, average='macro')



    # Create and plot confusion matrix
    cm = confusion_matrix(labels_1D, all_preds_1D)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix, Accuracy = {accuracy:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix {accuracy:.4f}.png')
    plt.show()

    print(f"\nOverall Test Results:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"f1_Score: {f1score:.4f}")
    return