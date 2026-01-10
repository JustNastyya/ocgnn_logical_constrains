# this file contains a json like form of all models and their training loops

from models.simple_ocgin.simple_ocgin import OCGIN, train_ocgin

model_reference = {
    "simple_ocgin": {
        "model": OCGIN,
        "train_loop": train_ocgin
    }
}