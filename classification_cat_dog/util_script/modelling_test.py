"""A scirpt to test modelling

"""
import torchvision

from classification_cat_dog.src import models

resnet18 = torchvision.models.resnet18(pretrained=True)
print(resnet18)

fc_head = models.FCHead([512, 100,  21])

resnet18.fc = fc_head
print(resnet18)

