import torch as T
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from models import InceptionModel


def evaluate_model(model_path,
                   features: np.ndarray,
                   labels: np.ndarray,
                   batch_size: int = 32  # New parameter to control batch size
                   ) -> Tuple[float, float]:
    # Initialize lists to store results across batches
    all_pred_labels = []
    all_true_labels = []
    total_loss = 0.0

    # Load the model
    model = InceptionModel(nb_classes=2)
    model.load_model(model_path)
    model.eval()  # Set to evaluation mode

    # Convert labels to tensor
    labels = T.tensor(labels, device=model.device, dtype=T.float32)

    # Process images in batches
    with T.no_grad():
        for i in range(0, len(features), batch_size):
            # Extract batch of images
            batch_features = features[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            # Preprocess batch images
            images = T.FloatTensor(batch_features.astype(np.float32) / 255.0)
            if len(images.shape) == 3:
                images = images.unsqueeze(1)
            images = images.to(model.device)

            # Forward pass
            predictions = model(images)

            # Calculate loss for this batch
            batch_loss = model.loss(batch_labels, predictions)
            total_loss += batch_loss.item()

            # Get predicted and true labels
            batch_pred_labels = T.argmax(predictions, dim=1).cpu().numpy()
            batch_true_labels = T.argmax(batch_labels, dim=1).cpu().numpy()

            # Collect results
            all_pred_labels.extend(batch_pred_labels)
            all_true_labels.extend(batch_true_labels)

    # Convert collected labels to numpy arrays
    all_pred_labels = np.array(all_pred_labels)
    all_true_labels = np.array(all_true_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    loss = total_loss / (len(features) // batch_size + 1)  # Average loss
    f1score = f1_score(all_true_labels, all_pred_labels, average='macro')
    roc_auc = roc_auc_score(all_true_labels, all_pred_labels, average='macro')

    # Create and plot confusion matrix
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix, Accuracy = {accuracy:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix {accuracy:.4f}.png')
    plt.show()
    plt.close()  # Close the plot to prevent memory issues

    print(f"\nOverall Test Results:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"f1_Score: {f1score:.4f}")
    print(f"roc_auc_score: {roc_auc:.4f}")

    return accuracy