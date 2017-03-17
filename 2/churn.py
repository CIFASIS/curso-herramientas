import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

cmap_bold = colors.ListedColormap(['#FF0000', '#00FF00'])
