import torch as T
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

def evaluate_model(net,
                   features: np.ndarray,
                   labels: np.ndarray,
                   ) -> Tuple[float, float]:
    """
    Evaluate the neural network performance using various metrics

    Args:
        net: Trained Naive_net model
        features: Input features
        labels: True labels
        test_split: Fraction of data to use for testing

    Returns:
        tuple of (accuracy, loss)
    """
    # Convert to tensors
    features = T.tensor(features, device=net.device, dtype=T.float32)
    labels = T.tensor(labels, device=net.device, dtype=T.float32)



    # Randomly shuffle indices



    # Evaluate model
    net.eval()  # Set to evaluation mode
    with T.no_grad():
        predictions = net(features)
        pred_labels = T.argmax(predictions, dim=1).cpu().numpy()
        labels_1D = T.argmax(labels, dim=1).cpu().numpy()


        # Calculate accuracy

        accuracy = np.mean(pred_labels == labels_1D)

        # Calculate loss
        n_classes = predictions.shape[1]
        print(predictions.shape, predictions.dtype)
        print(labels.shape, labels.dtype)
        loss = net.loss(predictions, labels).item()

        # Convert to numpy for sklearn metrics
        # pred_labels = pred_labels.cpu().numpy()
        true_labels = labels.cpu().numpy()
        predictions= predictions.cpu().numpy()

        # Print detailed classification report


        # Create and plot confusion matrix
        cm = confusion_matrix(labels_1D, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # Plot prediction distribution
        plt.figure(figsize=(10, 5))
        plt.hist(pred_labels, bins=n_classes, alpha=0.5, label='Predictions')
        plt.hist(true_labels, bins=n_classes, alpha=0.5, label='True Labels')
        plt.title('Distribution of Predictions vs True Labels')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.legend()
        plt.show()

    return accuracy, loss