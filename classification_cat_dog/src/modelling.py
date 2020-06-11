import torch
import torchvision


def build_model(out_feats):
    model = torchvision.models.resnet34(pretrained=True)
    fc_in_features = model.fc.in_features
    new_top = torch.nn.Sequential(
        torch.nn.Linear(fc_in_features, int((fc_in_features + out_feats)/2)),
        torch.nn.ReLU(),
        torch.nn.Linear(int((fc_in_features + out_feats)/2), out_feats),
        torch.nn.Softmax()
        )
    model.fc = new_top

    return model

