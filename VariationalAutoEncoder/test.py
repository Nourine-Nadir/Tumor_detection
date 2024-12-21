import torch as T
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from models import Naive_net, V_Encoder

def evaluate_model(net_model_path,
                   config,
                   latent_dim:int,
                   encoder_model_path,
                   images: np.ndarray,
                   labels: np.ndarray,

                   ) -> Tuple[float, float]:

    net = Naive_net(lr=config['lr'],
                    input_shape=config['latent_dim'],
                    fc1_dims=config['fc1_dims'],
                    fc2_dims=config['fc2_dims'],
                    fc3_dims=config['fc3_dims'],
                    fc4_dims=config['fc4_dims'],
                    n_output=labels.shape[-1])
    labels = T.tensor(labels, device=net.device, dtype=T.float32)
    encoder = V_Encoder(input_size= images.shape[-1],latent_dim=latent_dim)
    try:
        net.load_model(net_model_path)
        encoder.load_model(encoder_model_path, map_location=T.device('cuda'))

    except:
        print('Model could not be loaded for testing')
    # Convert to tensors


    with T.no_grad():  # Add this to prevent gradient computation for encoder
        images = T.FloatTensor(images.astype(np.float32) / 255.0)
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        images = images.to(encoder.device)
        print(f'images shape in training : {images.shape}')

        # Get encoded features
        encoded_features, _, _, _ = encoder(images)
        # Detach to prevent backward through encoder
        encoded_features = encoded_features.detach()






    # Evaluate model
    net.eval()  # Set to evaluation mode
    with T.no_grad():
        predictions = net(encoded_features)
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



    print(f"\nOverall Test Results:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"f1_Score: {f1score:.4f}")
    print(f"roc_auc_score: {roc_auc:.4f}")
    return accuracy