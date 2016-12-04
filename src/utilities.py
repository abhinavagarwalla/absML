from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import precision_recall_fscore_support
import os

__author__ = 'abhinav'

def _harm_mean(a, b, eps=1e-15):
    return (2*a*b/(a+b+eps))


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops = ['id', 'relevance', 'search_term', 'product_title', 'product_description', 'brand',
                       'product_info', 'attr']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches


class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions) ** 0.5
    return fmean_squared_error_


RMSE = make_scorer(fmean_squared_error, greater_is_better=False)


def change_to_int(params, indexes):
    for index in indexes:
        params[index] = int(params[index])
        # return params


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def identity(x):
    return x


class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return self.transformer(X)

# class ModelLoader(BaseEstimator, TransformerMixin):
#     def __init__(self, modelname, model):
#         pass

#     def fit(self):
#         pass

#     def transform(self, X, y)


def custom_scoring(y_pred, y):
    labels = sorted(list(set(y))[1:])
    score = precision_recall_fscore_support(y, y_pred, labels=labels, average='macro')
    score = _harm_mean(score[0], score[1])
    return score

