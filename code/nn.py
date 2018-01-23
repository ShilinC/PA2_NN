import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('./python-mnist/')
from mnist import MNIST

if __name__ == '__main__':

    data = MNIST('./python-mnist/data')
    training_images, training_labels = data.load_training()
    test_images, test_labels = data.load_testing()

    