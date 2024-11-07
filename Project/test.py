import torch as T
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

def evaluate_model(net,
                   features: np.ndarray,
                   labels: np.ndarray,
                   train_loss
                   ) -> Tuple[float, float]:

    # Convert to tensors
    features = T.tensor(features, device=net.device, dtype=T.float32)
    labels = T.tensor(labels, device=net.device, dtype=T.float32)






    # Evaluate model
    net.eval()  # Set to evaluation mode
    with T.no_grad():
        predictions = net(features)
        pred_labels = T.argmax(predictions, dim=1).cpu().numpy()
        labels_1D = T.argmax(labels, dim=1).cpu().numpy()


        # Calculate accuracy

        # accuracy = np.mean(pred_labels == labels_1D)
        accuracy = accuracy_score(pred_labels, labels_1D)

        # Calculate loss
        loss = net.loss(predictions, labels).item()


        # Print detailed classification report


        # Create and plot confusion matrix
        cm = confusion_matrix(labels_1D, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix, Accuracy = {accuracy:.4f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        plt.plot(train_loss)
        plt.show()



    return accuracy, loss