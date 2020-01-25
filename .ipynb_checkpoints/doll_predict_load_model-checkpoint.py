# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Keras
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping # コールバック関数
from keras.optimizers import SGD # 最適化関数
from keras.initializers import glorot_uniform, orthogonal, TruncatedNormal # 初期化関数
from keras.preprocessing.image import load_img, array_to_img, img_to_array # 画像の取り扱い
from keras.utils import np_utils, plot_model # ?
from keras.layers import Dense, Activation, Dropout, LSTM, Conv2D, MaxPooling2D, Flatten, BatchNormalization # レイヤー
from keras.layers.advanced_activations import PReLU # 応用活性化関数
from keras.layers.recurrent import GRU, SimpleRNN # RNN系関数
from keras import backend as K

# ファインチューニング
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2

# 計算
import pandas as pd #行列計算
import numpy as np #行列計算
import math #数値計算
import itertools #順列・組み合わせ

# 図・画像
import matplotlib.pyplot as plt #グラフ
import seaborn as sns
from PIL import Image, ImageFilter

# 金融
import mpl_finance as mpf
import talib as ta # テクニカル指標

# scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, VotingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error


# GBDT
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
#from catboost import CatBoostClassifier, Pool

# フォント
import matplotlib.font_manager as fm
from matplotlib import rcParams

# その他
import glob
import re
import gc
import cv2
import os
from datetime import datetime, timedelta

# マルチプロセス・マルチスレッド
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# -

BATCH_SIZE = 512
EPOCHS = 100
VERBOSE = 1
VALIDATION_SPLIT = 0.2
MODEL = 'CNN'#'ResNet50'#'CNN' #'MyVGG16'
IMG_CHANNELS = 3
#KAITEN_MIZUMASHI = #range(0, 360, 30) # range(0, 1, 90)
KAITEN_MIZUMASHI = range(0, 1, 90)


# 余白を追加
def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


# 余白を追加して正方形に変形
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


# +
img_size = 128
X = []
y = []

#img_path_list_1 = glob.glob('image/ddh-01/*')
#img_path_list_2 = glob.glob('image/ddh-10/*')
#img_path_list_1 = glob.glob('image/ddh-01/*')
#img_path_list_2 = glob.glob('image/ddh-10/*')

img_path_list_1 = glob.glob('image/ddh-01_under_50000/*')
img_path_list_2 = glob.glob('image/ddh-10_under_50000/*')
img_path_list_3 = glob.glob('image/ddh-06_under_50000/*')
img_path_list_4 = glob.glob('image/ddh-07_under_50000/*')
img_path_list_5 = glob.glob('image/ddh-09_under_50000/*')

#img_path_list_1 = glob.glob('image/ddh-01_face_under_50000/*')
#img_path_list_2 = glob.glob('image/ddh-10_face_under_50000/*')
img_path_list = img_path_list_1# + img_path_list_2# + img_path_list_3 + img_path_list_4

for path in img_path_list[:100]:
    print(path)
    img = Image.open(path) # 開いて
    img = expand2square(img, (255, 255, 255)) # 正方形に整形して
    img = img.resize((img_size, img_size)) # サイズを調整する
    
    # グレースケールに変換
    if IMG_CHANNELS == 1:
        img = img.convert('L')
        img = np.array(img)
        img = img.reshape(img_size, img_size, 1) # データの形式を変更
        X.append(img) # 学習データ
    else:
        X.append(np.array(img)) # 学習データ
        
    y.append(int(re.search(r'[0-9]{8,8}', path).group(0))) # ラベルデータ
    
# numpy配列に変換
X = np.array(X)
y = np.array(y)
# -

# numpy配列に変換
X_ = np.array(X)
y_ = np.array(y)

model = load_model('model_doll_cnn', compile=False)

# +
pred = model.predict(X_).flatten()

df_pred = pd.DataFrame([pred, y_])
df_pred = df_pred.T
df_pred.to_csv("pred.csv", index=False)

df_pred
# -






