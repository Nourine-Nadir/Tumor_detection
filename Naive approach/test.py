import torch as T
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from models import Naive_net

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

        accuracy = accuracy_score(labels_1D, pred_labels)

        # Calculate loss
        loss = net.loss(labels, predictions).item()

        # Calculate f1score
        f1score = f1_score(labels_1D, pred_labels, average='macro')

        # Calculate roc_auc_score
        roc_auc = roc_auc_score(labels_1D, pred_labels, average='macro')



        # Create and plot confusion matrix
        cm = confusion_matrix(labels_1D, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix, Accuracy = {accuracy:.4f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix 0.9807.png')
        plt.show()

        plt.plot(train_loss)
        plt.savefig('loss.png')
        plt.show()

    print(f"\nOverall Test Results:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"f1_Score: {f1score:.4f}")
    print(f"roc_auc_score: {roc_auc:.4f}")
    return accuracy