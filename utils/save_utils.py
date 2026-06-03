import pickle
import torch


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def save_label_encoder(le, path):
    with open(path, "wb") as f:
        pickle.dump(le, f)


def load_label_encoder(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_report(report, path):
    with open(path, "w") as f:
        f.write(report)

