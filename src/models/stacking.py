from tree import dtrees
from xgb import xgbs
from rfr import rfrs
from etree import etrees
from gbc import gbcs

models = {}

models.update(dtrees)
models.update(xgbs)
models.update(rfrs)
models.update(etrees)
models.update(gbcs)


level0 = models.items()