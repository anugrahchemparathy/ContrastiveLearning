import torch
import math
import sys

if 'sklearnex' in sys.modules:
    from sklearnex import patch_sklearn
    patch_sklearn()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from .device import t2np

def get_entr(model, traind, testd, device=torch.device('cpu'), prop=1.0, swapc=False):
    model.eval()
    with torch.no_grad():
        train_en = []
        train_tr = []
        for it, (input1, input2, y) in enumerate(traind):
            if it > len(traind) * prop:
                break
            input1 = input1.type(torch.float32).to(device)
            if swapc:
                input1 = torch.swapaxes(input1, 1, 3)
            train_en.append(model.encoder(input1))
            train_tr.append(y)
        train_en = torch.cat(train_en, axis=0)[:math.floor(len(traind.dataset) * prop)]
        train_tr = torch.cat(train_tr)[:math.floor(len(traind.dataset) * prop)]

        test_en = []
        test_tr = []
        for it, (input1, input2, y) in enumerate(testd):
            if it > len(testd) * prop:
                break
            input1 = input1.type(torch.float32).to(device)
            if swapc:
                input1 = torch.swapaxes(input1, 1, 3)
            test_en.append(model.encoder(input1))
            test_tr.append(y)
        test_en = torch.cat(test_en, axis=0)[:math.floor(len(testd.dataset) * prop)]
        test_tr = torch.cat(test_tr)[:math.floor(len(testd.dataset) * prop)]
    return train_en, train_tr, test_en, test_tr

def knn_eval(model, traind, testd, k=3, device=torch.device('cpu'), prop=1.0, swapc=False):
    train_en, train_tr, test_en, test_tr = get_entr(model, traind, testd, device=device, prop=prop, swapc=swapc)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(t2np(train_en), t2np(train_tr))
    return knn.score(t2np(test_en), t2np(test_tr))

def lin_eval(model, traind, testd, device=torch.device('cpu'), prop=1.0, swapc=False):
    train_en, train_tr, test_en, test_tr = get_entr(model, traind, testd, device=device, prop=prop, swapc=swapc)

    train_en = train_en / (1e-10 + torch.linalg.vector_norm(train_en, dim=1, keepdim=True))
    test_en = test_en / (1e-10 + torch.linalg.vector_norm(test_en, dim=1, keepdim=True))
    train_en = train_en - torch.mean(train_en, dim=0, keepdim=True)
    test_en = test_en - torch.mean(test_en, dim=0, keepdim=True)
    train_en = train_en / (1e-10 + torch.std(train_en, (0,), True, keepdim=True))
    test_en = test_en / (1e-10 + torch.std(test_en, (0,), True, keepdim=True))

    lr = LogisticRegression()
    lr.fit(t2np(train_en), t2np(train_tr))
    return lr.score(t2np(test_en), t2np(test_tr))
