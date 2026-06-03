from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch



def evaluate_model(model, X_test_t, device):
    model.eval()

    with torch.no_grad():
        logits = model(X_test_t.to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    return preds