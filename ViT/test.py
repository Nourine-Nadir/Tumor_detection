import torch as T
import numpy as np
from vit import ViT
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
    DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
    with T.no_grad():  # Add this to prevent gradient computation for encoder
        images = T.FloatTensor(np.array(images).astype(np.float32) / 255.0)
    if len(images.shape) == 3 :
        images = T.unsqueeze(images, dim=1)
    print(images.shape)


    labels = T.tensor(np.argmax(np.array(labels), axis=-1), device=DEVICE, dtype=T.long)
    # Or if labels are already 1D



    criterion = T.nn.CrossEntropyLoss()
    with T.no_grad():
        predictions = model(images)
        pred_labels = T.argmax(predictions, dim=1).cpu().numpy()
        labels_1D = labels.cpu().numpy()
        print(predictions.shape)
        # Calculate accuracy

        accuracy = accuracy_score(labels_1D, pred_labels)

        # Calculate loss
        loss = criterion(predictions, labels).item()

        # Calculate f1score
        f1score = f1_score(labels_1D, pred_labels, average='macro')

        # Calculate roc_auc_score
        # roc_auc = roc_auc_score(labels_1D, pred_labels, average='macro')

        # Create and plot confusion matrix
        cm = confusion_matrix(labels_1D, pred_labels)
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

    model.save_model(model_path+ ' last_model')
    return