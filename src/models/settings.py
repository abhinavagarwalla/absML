import os,sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,parentdir)
configs = {}
from configs import *