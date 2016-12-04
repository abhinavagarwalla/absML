from utilities import SimpleTransform
from generate_dataset import generate_dataset
import numpy as np

__author__ = 'abhinav'


def log_transform(x):
    return np.log(1 + x)


if __name__ == '__main__':
    features = [
        ('mainFeatures_log10_normalized', SimpleTransform()),
        # ('keras_feature', SimpleTransform()),

    ]

    generate_dataset(features, 'main_normalized_log10')