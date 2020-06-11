"""A scirpt to test modelling

"""
from classification_cat_dog.src import modelling


model = modelling.build_model(10)
print(model)