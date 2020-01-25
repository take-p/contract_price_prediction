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

# %%HTML
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 95%; }
    div#maintoolbar-container { width: 99%; }
</style>

BATCH_SIZE = 512
EPOCHS = 100
VERBOSE = 1
VALIDATION_SPLIT = 0.2
MODEL = 'CNN'#'ResNet50'#'CNN' #'MyVGG16'
IMG_CHANNELS = 3
#KAITEN_MIZUMASHI = #range(0, 360, 30) # range(0, 1, 90)
KAITEN_MIZUMASHI = range(0, 1, 90)

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


# ## ドール画像の回帰分析

# ### 画像を読み込む

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
img_path_list_3 = glob.glob('image/ddh-07_under_50000/*')
img_path_list_4 = glob.glob('image/ddh-09_under_50000/*')
#img_path_list_1 = glob.glob('image/ddh-01_face_under_50000/*')
#img_path_list_2 = glob.glob('image/ddh-10_face_under_50000/*')
img_path_list = img_path_list_1 + img_path_list_2 + img_path_list_3 + img_path_list_4

for path in img_path_list:
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

# ## データの分析

print(y)
sns.distplot(y, kde=False, rug=False, bins=50)
#sns.distplot(y)

# ### データを訓練データと評価データとテストデータに分割

"""
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=0
)

print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)
"""

# +
#"""
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=0
)

"""
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    shuffle=True,
    random_state=0
)
"""

print("X_train", X_train.shape)
print("y_train", y_train.shape)
#print("X_valid", X_valid.shape)
#print("y_valid", y_valid.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)
#"""
# -

# ## 学習データの水増し

# +
X_train_extend = []
y_train_extend = []

for X_, y_, j in zip(X_train, y_train, range(len(X_train))):
    print("X_train[", j, "]")
    
    if IMG_CHANNELS == 1:
        X_ = X_.reshape(img_size, img_size)
        img = Image.fromarray(X_)
    else:
        img = Image.fromarray(X_) # numpy配列から画像に戻す。
    img_mirror = img.transpose(Image.FLIP_LEFT_RIGHT) # 左右反転
    
    # 回転
    for img_tmp in [img, img_mirror]:
        for i in KAITEN_MIZUMASHI:
            #print("角度", i, "°")
            img_tmp2 = img_tmp.rotate(i)
            #print(img_tmp.shape)
            #print(X_train.shape)
            if IMG_CHANNELS == 1:
                img_tmp2 = np.array(img_tmp2)
                img_tmp2 = img_tmp2.reshape(img_size, img_size, 1)
                X_train_extend.append(img_tmp2)
                y_train_extend.append(y_)
            else:
                X_train_extend.append(np.array(img_tmp2))
                y_train_extend.append(y_)
        
            # フィルター
            #"""
            #img_tmp3 = img_tmp2.filter(ImageFilter.FIND_EDGES)
            #X_train_extend.append(np.array(img_tmp3))
            #y_train_extend.append(y_)
            """
            img_tmp3 = img_tmp2.filter(ImageFilter.EDGE_ENHANCE)
            X_train_extend.append(np.array(img_tmp3))
            y_train_extend.append(y_)
            img_tmp3 = img_tmp2.filter(ImageFilter.EDGE_ENHANCE_MORE)
            X_train_extend.append(np.array(img_tmp3))
            y_train_extend.append(y_)
            img_tmp3 = img_tmp2.filter(ImageFilter.UnsharpMask(radius=5, percent=150, threshold=2))
            X_train_extend.append(np.array(img_tmp3))
            y_train_extend.append(y_)
            img_tmp3 = img_tmp2.filter(ImageFilter.UnsharpMask(radius=10, percent=200, threshold=5))
            X_train_extend.append(np.array(img_tmp3))
            y_train_extend.append(y_)
            """

# +
# numpy配列に変換
X_train_extend = np.array(X_train_extend)
y_train_extend = np.array(y_train_extend)

print("X_train_extend", X_train_extend.shape)
print("y_train_extend", y_train_extend.shape)
#print("X_valid", X_valid.shape)
#print("y_valid", y_valid.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)
# -

# データを保存
"""
np.save('X_train_extend', X_train_extend)
np.save('y_train_extend', y_train_extend)


X_train_extend = np.load(file="X_train_extend.npy")
y_train_extend = np.load(file="y_train_extend.npy")
"""


# +
# データを正規化（標準化は平均が0、標準偏差が1になるようにデータを加工すること。正規化は最低が0、最高が2になるようにデータを加工すること。）
#X_train_extend = X_train_extend / 255
#X_valid = X_valid / 255
#X_test = X_test / 255
# -

# ## モデル構築

def create_model(model_type, img_width, img_height, img_channels):
    model = Sequential()
    
    if model_type == 'CNN':
        model.add(Conv2D(
            filters = 16,
            kernel_size = (2, 2),
            strides = 1,
            padding = 'same',
            input_shape=(img_height, img_width, img_channels)))
        model.add(PReLU())
        model.add(Dropout(0.2))
        
        """
        model.add(Conv2D(
            filters = 16,
            kernel_size = (1, 2),
            strides = 1,
            padding = 'same',
            activation = 'relu',
            input_shape=(img_height, img_width, img_channels)))
        
        model.add(Conv2D(
            filters = 16,
            kernel_size = (2, 1),
            strides = 1,
            padding = 'same',
            activation = 'relu',
            input_shape=(img_height, img_width, img_channels)))
        """
        
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        ##model.add(Dropout(0.2))
        
        model.add(Conv2D(
            filters = 32,
            kernel_size = (2, 2),
            strides = 1,
            padding = 'same'))
        model.add(PReLU())
        #model.add(Dropout(0.2))
        
        """
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.25))
        
        model.add(Conv2D(
            filters = 64,
            kernel_size = (3, 3),
            strides = 1,
            padding = 'same',
            activation = 'relu'))
        """
        
        #model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(32))
        model.add(PReLU())
        #model.add(Dropout(0.2))
        model.add(Dense(1))
        model.add(PReLU())
        model.compile(
            loss='mse',
            optimizer='Adam',
            metrics=['mae'])
        
    elif model_type == 'CNN2':    
        # 畳み込み層
        model.add(Conv2D(
            filters = 16, # フィルターの枚数
            kernel_size = (3, 3), # フィルターのサイズ
            strides = 1, # フィルターの移動幅
            padding='same', # 画像の周りを0パディングすることで端の特徴を捉えられるようにする
            activation='relu', # 活性化関数
            input_shape=(img_height, img_width, img_channels))) # 画像のサイズ
        
        # 畳み込み層
        model.add(Conv2D(
            filters = 16, # フィルターの枚数
            kernel_size = (3, 3), # フィルターのサイズ
            strides = 1, # フィルターの移動幅
            padding='same', # 画像の周りを0パディングすることで端の特徴を捉えられるようにする
            activation='relu', # 活性化関数
            input_shape=(img_height, img_width, img_channels))) # 画像のサイズ
    
        # プーリング層（情報量の削減）
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # ドロップアウト層
        model.add(Dropout(0.25))
        
        #"""
        # 畳み込み層
        model.add(Conv2D(
            filters = 32,
            kernel_size = (3, 3),
            strides = 1,
            padding='same',
            activation='relu'))
        
        # 畳み込み層
        model.add(Conv2D(
            filters = 32,
            kernel_size = (3, 3),
            strides = 1,
            padding='same',
            activation='relu'))
        
        # プーリング層
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # ドロップアウト層
        model.add(Dropout(0.25))
        #"""
        
        # 多層パーセプトロンに入力するにあたり４次元配列を１次元配列に変換
        model.add(Flatten())
        
        # 全結合層
        #model.add(Dense(512, activation='relu'))
        
        # ドロップアウト層
        #model.add(Dropout(0.5))
        
        # 全結合層
        model.add(Dense(1, activation='relu'))
        
        # 活性化層
        #model.add(Activation('softmax'))
        #model.add(Activation('relu'))
        
        # コンパイル
        model.compile(
            loss='mse',
            optimizer='Adam',
            metrics=['mae'])
        
    elif model_type == 'MyVGG16':
        # 畳み込み層
        model.add(Conv2D(
            32, (3, 3), # フィルタ数とフィルタサイズ
            padding='same', # 0パディング
            activation='relu', # 活性化関数
            input_shape=(img_height, img_width, img_channels))) # 画像のサイズ
        
        # 畳み込み層
        model.add(Conv2D(
            32, (3, 3), # 16は？。(3, 3)はフィルタのサイズ
            padding='same', # ?
            activation='relu', # 活性化関数
            input_shape=(img_height, img_width, img_channels))) # 画像のサイズ
        
        # 畳み込み層
        model.add(Conv2D(
            32, (3, 3), # 16は？。(3, 3)はフィルタのサイズ
            padding='same', # ?
            activation='relu', # 活性化関数
            input_shape=(img_height, img_width, img_channels))) # 画像のサイズ
        
        # 畳み込み層
        #model.add(Conv2D(
        #    16, (3, 3), # 16は？。(3, 3)はフィルタのサイズ
        #    padding='same', # ?
        #    activation='relu', # 活性化関数
        #    input_shape=(img_height, img_width, img_channels))) # 画像のサイズ
    
        # プーリング層（情報量の削減）基本は(2, 2)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # ドロップアウト層
        model.add(Dropout(0.25))
        
        
        # 畳み込み層
        model.add(Conv2D(
            32, (3, 3),
            padding='same',
            activation='relu'))
        
        # 畳み込み層
        model.add(Conv2D(
            32, (3, 3),
            padding='same',
            activation='relu'))
        
        # 畳み込み層
        model.add(Conv2D(
            32, (3, 3),
            padding='same',
            activation='relu'))
        
        # 畳み込み層
        #model.add(Conv2D(
        #    32, (3, 3),
        #    padding='same',
        #    activation='relu'))
        
        # プーリング層
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # ドロップアウト層
        model.add(Dropout(0.25))
        
        # 多層パーセプトロンに入力するにあたり４次元配列を１次元配列に変換
        model.add(Flatten())
        
        # 全結合層
        model.add(Dense(512, activation='relu'))
        
        # ドロップアウト層
        model.add(Dropout(0.5))
        
        # 全結合層
        model.add(Dense(1))
        
        # 活性化層
        #model.add(Activation('softmax'))
        model.add(Activation('relu'))
        
        # コンパイル
        model.compile(
            loss='mse',
            optimizer='Adam',
            metrics=['mae'])
        
    elif model_type == 'VGG16':
        # VGG16
        model = VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=(img_width, img_height, img_channels),
            #pooling=None,
            classes=1,
        )
        
        
        # 全結合層
        #model.add(Dense(NUM_CLASSES))
        
        # 活性化層
        #model.add(Activation('softmax'))
        
        # コンパイル
        model.compile(
            loss='mse',
            optimizer='Adam',
            metrics=['mae'])
        
    elif model_type == 'VGG19':
        model = VGG19(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=(img_width, img_height, img_channels),
            #pooling=None,
            classes=1
        )
        
        
        # コンパイル
        model.compile(
            loss='mse',
            optimizer='Adam',
            metrics=['mae'])
        
    elif model_type == 'ResNet50':
        model = ResNet50(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=(img_width, img_height, img_channels),
            #pooling=None,
            classes=1
        )
        
        # コンパイル
        model.compile(
            loss='mse',
            optimizer='Adam',
            metrics=['mae'])
        
    elif model_type == 'InceptionResNetV2':
        model = InceptionResNetV2(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=(img_width, img_height, img_channels),
            #pooling=None,
            classes=2
        )
        
        # コンパイル
        model.compile(
            loss='mse',
            optimizer='Adam',
            metrics=['mae'])
        
    return model

# モデルの作成
model = create_model(MODEL, img_size, img_size, IMG_CHANNELS)
model.summary()

# +
early_stopping = EarlyStopping(patience=5, verbose=1)

# 学習（戻り値はログ）
hist = model.fit(
    X_train_extend,
    y_train_extend,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks = [early_stopping],
    verbose=VERBOSE, # 出力の詳細
    #validation_data=(X_valid, y_valid)#,自前で用意した評価データを使用
    shuffle = True,
    validation_split=VALIDATION_SPLIT
)
# -

score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
print('MAE(平均絶対誤差)', score[1])
print('MSE(平均二乗誤差)', score[0])
print('RMSE(平方平均二乗誤差)', np.sqrt(score[0]))

# +
pred = model.predict(X_test).flatten()

df_pred = pd.DataFrame([pred, y_test])
df_pred = df_pred.T
df_pred.to_csv("pred.csv", index=False)

df_pred

# +
#学習の様子をグラフへ描画
#正解率の推移をプロット
plt.plot(hist.history['mean_absolute_error'])
plt.plot(hist.history['val_mean_absolute_error'])
plt.title('MAE')
plt.legend(['train','valid'],loc='upper left')
plt.show()

#ロスの推移をプロット
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['train','valid'],loc='upper left')
plt.show()
# -

# # フィルターを可視化

# +
from keras.models import load_model
import numpy as np
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img
import cv2
from IPython.display import display_png

#モデルの読込み
#file_name='vgg16_madomagi_SGD0.001_G_D4096_D512'
#model=load_model('./result/' + file_name+'.h5')
model.summary()

begin = 0
end = 100
for i, p, y in zip(X_test[begin:end], pred[begin:end], y_test[begin:end]):
    print('predict: ', p, '円')
    print('test: ', y, '円')
    
    #対象イメージの読込み
    #jpg_name = 'madoka0137'
    #img_path = ('./madoka_magica_images/display/' + jpg_name + '.jpg')
    #img_path = 'image/ddh-01_aucfree_custom_head_only_under_50000/00019000_ece722b6-c39c-4a01-b34f-951f72dcd62e.jpeg'
    #img = img_to_array(load_img(img_path, target_size=(224,224)))
    img = i
    H,W =img.shape[:2]
    img_nad = img_to_array(img)/255
    img_nad = img_nad[None, ...]
    
    for j in [2]:
        #特長マップを抜き出すレイヤー指定
        get_layer_output = K.function([model.layers[0].input],[model.layers[j].output])
        print(model.layers[j])
        layer_output = get_layer_output([img_nad])[0]

        #特長マップ合成
        G, R, ch = layer_output.shape[1:]
        res = np.zeros((G,R))

        for i in range(ch):
            img_res = layer_output[0,:,:,i]
            res = res + img_res       

        res = res/ch

        #特長マップ平均の平坦化
        res_flatte = np.ma.masked_equal(res,0)
        res_flatte = (res_flatte - res_flatte.min())*255/(res_flatte.max()-res_flatte.min())
        res_flatte = np.ma.filled(res_flatte,0)

        #色付け
        acm_img = cv2.applyColorMap(np.uint8(res_flatte), cv2.COLORMAP_JET)
        acm_img = cv2.cvtColor(acm_img, cv2.COLOR_BGR2RGB)
        acm_img = cv2.resize(acm_img,(H,W))

        #元絵と合成
        mixed_img = (np.float32(acm_img)*0.6 + img *0.4)

        #表示
        out_img = np.concatenate((img, acm_img, mixed_img), axis=1)
        display_png(array_to_img(out_img))
# -

# # K分割交差検証

# +
kf = KFold(n_splits=5, shuffle=True)
early_stopping = EarlyStopping(patience=5, verbose=1)
scores = []
preds = []

for train_index, eval_index in kf.split(X_train_extend):
    X_tra, X_eval = X_train_extend[train_index], X_train_extend[eval_index]
    y_tra, y_eval = y_train_extend[train_index], y_train_extend[eval_index]
    
    model = create_model(MODEL, img_size, img_size, IMG_CHANNELS)
    
    hist = model.fit(
        X_tra,
        y_tra,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks = [early_stopping],
        verbose=VERBOSE,
        validation_data=(X_eval, y_eval)
    )
    
    scores.append(model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1))
    preds.append(model.predict(X_test).flatten())

# +
for score in scores:
    print("MAE(平均絶対誤差):", score[1])
    print("MSE(平均二乗誤差):", score[0])
    print("MAE(平均絶対誤差):", np.sqrt(score[0]), '\n')

print('平均')
score = np.mean(np.array(scores), axis=0)
print('MAE(平均絶対誤差)', score[1])
print('MSE(平均二乗誤差)', score[0])
print('RMSE(平方平均二乗誤差)', np.sqrt(score[0]))

# +
##### 
tmp = np.array(preds)
pred = np.mean(tmp, axis=0)
#print(tmp)
df_pred = pd.DataFrame([pred, y_test]).T
df_pred.columns = ['予測', '実際の価格']

df_kf = pd.DataFrame(tmp).T
df_kf = df_kf.join(df_pred)

df_kf.to_csv('pred_kf.csv')
df_kf
# -

for img, i in zip(X_test, range(len(df_kf))):
    print(df_kf[df_kf.index==i])
    display_png(array_to_img(img))

# # 全体画像と顔画像での比較

# +
img_size = 128
X_1 = []
y_1 = []
X_2 = []
y_2 = []

ddh = ['01', '10', '06', '07', '09']

for d in ddh:
    img_path_list = os.listdir('image/ddh-'+d+'_face_under_50000')
    for path in img_path_list:
        print(path)
        if os.path.exists('image/ddh-'+d+'_under_50000/'+path):
            img1 = Image.open('image/ddh-'+d+'_under_50000/'+path) # 開いて
            img1 = expand2square(img1, (255, 255, 255)) # 正方形に整形して
            img1 = img1.resize((img_size, img_size)) # サイズを調整する
            img2 = Image.open('image/ddh-'+d+'_face_under_50000/'+path) # 開いて
            img2 = expand2square(img2, (255, 255, 255)) # 正方形に整形して
            img2 = img2.resize((img_size, img_size)) # サイズを調整する
            X_1.append(np.array(img1))
            y_1.append(int(re.search(r'[0-9]{8,8}', 'image/ddh-'+d+'_under_50000/'+path).group(0)))
            X_2.append(np.array(img2))
            y_2.append(int(re.search(r'[0-9]{8,8}', 'image/ddh-'+d+'_face_under_50000/'+path).group(0)))

    img_path_list = os.listdir('image/ddh-'+d+'_face_under_50000')
    
# numpy配列に変換
X_1 = np.array(X_1)
y_1 = np.array(y_1)
X_2 = np.array(X_2)
y_2 = np.array(y_2)

# +
#"""
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    X_1,
    y_1,
    test_size=0.2,
    shuffle=True,
    random_state=0
)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_2,
    y_2,
    test_size=0.2,
    shuffle=True,
    random_state=0
)

"""
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    shuffle=True,
    random_state=0
)
"""

print("X_train_1", X_train_1.shape)
print("y_train_1", y_train_1.shape)
print("X_train_2", X_train_2.shape)
print("y_train_2", y_train_2.shape)
#print("X_valid", X_valid.shape)
#print("y_valid", y_valid.shape)
print("X_test_1", X_test_1.shape)
print("y_test_1", y_test_1.shape)
print("X_test_2", X_test_2.shape)
print("y_test_2", y_test_2.shape)
#"""

# +
X_train_extend_1 = []
y_train_extend_1 = []
X_train_extend_2 = []
y_train_extend_2 = []

for X_, y_, j in zip(X_train_1, y_train_1, range(len(X_train_1))):
    print("X_train[", j, "]")
    
    if IMG_CHANNELS == 1:
        X_ = X_.reshape(img_size, img_size)
        img = Image.fromarray(X_)
    else:
        img = Image.fromarray(X_) # numpy配列から画像に戻す。
    img_mirror = img.transpose(Image.FLIP_LEFT_RIGHT) # 左右反転
    
    # 回転
    for img_tmp in [img, img_mirror]:
        for i in KAITEN_MIZUMASHI:
            #print("角度", i, "°")
            img_tmp2 = img_tmp.rotate(i)
            #print(img_tmp.shape)
            #print(X_train.shape)
            if IMG_CHANNELS == 1:
                img_tmp2 = np.array(img_tmp2)
                img_tmp2 = img_tmp2.reshape(img_size, img_size, 1)
                X_train_extend_1.append(img_tmp2)
                y_train_extend_1.append(y_)
            else:
                X_train_extend_1.append(np.array(img_tmp2))
                y_train_extend_1.append(y_)
                
for X_, y_, j in zip(X_train_2, y_train_2, range(len(X_train_2))):
    print("X_train[", j, "]")
    
    if IMG_CHANNELS == 1:
        X_ = X_.reshape(img_size, img_size)
        img = Image.fromarray(X_)
    else:
        img = Image.fromarray(X_) # numpy配列から画像に戻す。
    img_mirror = img.transpose(Image.FLIP_LEFT_RIGHT) # 左右反転
    
    # 回転
    for img_tmp in [img, img_mirror]:
        for i in KAITEN_MIZUMASHI:
            #print("角度", i, "°")
            img_tmp2 = img_tmp.rotate(i)
            #print(img_tmp.shape)
            #print(X_train.shape)
            if IMG_CHANNELS == 1:
                img_tmp2 = np.array(img_tmp2)
                img_tmp2 = img_tmp2.reshape(img_size, img_size, 1)
                X_train_extend_2.append(img_tmp2)
                y_train_extend_2.append(y_)
            else:
                X_train_extend_2.append(np.array(img_tmp2))
                y_train_extend_2.append(y_)
                
# numpy配列に変換
X_train_extend_1 = np.array(X_train_extend_1)
y_train_extend_1 = np.array(y_train_extend_1)
X_train_extend_2 = np.array(X_train_extend_2)
y_train_extend_2 = np.array(y_train_extend_2)
# -

print("X_train_extend_1", X_train_extend_1.shape)
print("y_train_extend_1", y_train_extend_1.shape)
print("X_train_extend_2", X_train_extend_2.shape)
print("y_train_extend_2", y_train_extend_2.shape)

# +
early_stopping = EarlyStopping(patience=10, verbose=1)

# モデルの作成
model_1 = create_model(MODEL, img_size, img_size, IMG_CHANNELS)
model_1.summary()

# 学習（戻り値はログ）
hist_1 = model_1.fit(
    X_train_extend_1,
    y_train_extend_1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks = [early_stopping],
    verbose=VERBOSE, # 出力の詳細
    #validation_data=(X_valid, y_valid)#,自前で用意した評価データを使用
    shuffle = True,
    validation_split=VALIDATION_SPLIT
)

# モデルの作成
model_2 = create_model(MODEL, img_size, img_size, IMG_CHANNELS)
model_2.summary()

# 学習（戻り値はログ）
hist_2 = model_2.fit(
    X_train_extend_2,
    y_train_extend_2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks = [early_stopping],
    verbose=VERBOSE, # 出力の詳細
    #validation_data=(X_valid, y_valid)#,自前で用意した評価データを使用
    shuffle = True,
    validation_split=VALIDATION_SPLIT
)

# +
pred_1 = model_1.predict(X_test_1).flatten()
pred_2 = model_2.predict(X_test_2).flatten()
df_pred_mean = (pred_1+pred_2)/2

df_pred_predict = pd.DataFrame([pred_1, pred_2, df_pred_mean, y_test_1])
df_pred_predict = df_pred_predict.T
df_pred_predict.columns = ['全体', '顔', '平均', '実際の値']
df_pred_predict.to_csv("pred_predict.csv", index=False)

df_pred_predict

# +
print("全身")
score_1 = model_1.evaluate(X_test_1, y_test_1, batch_size=BATCH_SIZE, verbose=1)
print('MAE(平均絶対誤差)', score_1[1])
print('MSE(平均二乗誤差)', score_1[0])
print('RMSE(平方平均二乗誤差)', np.sqrt(score_1[0]))

print("\n顔")
score_2 = model_2.evaluate(X_test_2, y_test_2, batch_size=BATCH_SIZE, verbose=1)
print('MAE(平均絶対誤差)', score_2[1])
print('MSE(平均二乗誤差)', score_2[0])
print('RMSE(平方平均二乗誤差)', np.sqrt(score_2[0]))

print("\n全身と顔の平均")
print('MAE()平均絶対誤差', mean_absolute_error(df_pred_mean, y_test_1))
print('RMSE(平均二乗誤差)', mean_squared_error(df_pred_mean, y_test_1))
print('RMSE(平方平均二乗誤差)', np.sqrt(mean_squared_error(df_pred_mean, y_test_1)))

# +
#学習の様子をグラフへ描画
#正解率の推移をプロット
plt.plot(hist_1.history['mean_absolute_error'])
plt.plot(hist_1.history['val_mean_absolute_error'])
plt.title('MAE')
plt.legend(['train','valid'],loc='upper left')
plt.show()

#ロスの推移をプロット
plt.plot(hist_1.history['loss'])
plt.plot(hist_1.history['val_loss'])
plt.title('Loss')
plt.legend(['train','valid'],loc='upper left')
plt.show()

#学習の様子をグラフへ描画
#正解率の推移をプロット
plt.plot(hist_2.history['mean_absolute_error'])
plt.plot(hist_2.history['val_mean_absolute_error'])
plt.title('MAE')
plt.legend(['train','valid'],loc='upper left')
plt.show()

#ロスの推移をプロット
plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])
plt.title('Loss')
plt.legend(['train','valid'],loc='upper left')
plt.show()
# -

begin = 0
end = 100
for i, j, k in zip(X_test_1[begin:end], X_test_2[begin:end], range(len(df_pred_predict))):
    print(df_pred_predict[df_pred_predict.index==k])
    
    img_1 = i
    img_2 = j
    H_1,W_1 =img_1.shape[:2]
    H_2,W_2 =img_2.shape[:2]
    img_nad_1 = img_to_array(img_1)/255
    img_nad_2 = img_to_array(img_2)/255
    img_nad_1 = img_nad_1[None, ...]
    img_nad_2 = img_nad_2[None, ...]
    
    for layer in [2]:
        #特長マップを抜き出すレイヤー指定
        get_layer_output_1 = K.function([model_1.layers[0].input],[model_1.layers[layer].output])
        get_layer_output_2 = K.function([model_2.layers[0].input],[model_2.layers[layer].output])
        print(model_1.layers[layer])
        print(model_2.layers[layer])
        layer_output_1 = get_layer_output_1([img_nad_1])[0]
        layer_output_2 = get_layer_output_2([img_nad_2])[0]

        #特長マップ合成
        G_1, R_1, ch_1 = layer_output_1.shape[1:]
        G_2, R_2, ch_2 = layer_output_2.shape[1:]
        res_1 = np.zeros((G_1,R_1))
        res_2 = np.zeros((G_2,R_2))

        for i in range(ch_1):
            img_res_1 = layer_output_1[0,:,:,i]
            res_1 = res_1 + img_res_1

        res_1 = res_1/ch_1
        
        for i in range(ch_2):
            img_res_2 = layer_output_2[0,:,:,i]
            res_2 = res_2 + img_res_2
            
        res_2 = res_2/ch_2

        #特長マップ平均の平坦化
        res_flatte_1 = np.ma.masked_equal(res_1,0)
        res_flatte_2 = np.ma.masked_equal(res_2,0)
        res_flatte_1 = (res_flatte_1 - res_flatte_1.min())*255/(res_flatte_1.max()-res_flatte_1.min())
        res_flatte_2 = (res_flatte_2 - res_flatte_2.min())*255/(res_flatte_2.max()-res_flatte_2.min())
        res_flatte_1 = np.ma.filled(res_flatte_1,0)
        res_flatte_2 = np.ma.filled(res_flatte_2,0)

        #色付け
        acm_img_1 = cv2.applyColorMap(np.uint8(res_flatte_1), cv2.COLORMAP_JET)
        acm_img_2 = cv2.applyColorMap(np.uint8(res_flatte_2), cv2.COLORMAP_JET)
        acm_img_1 = cv2.cvtColor(acm_img_1, cv2.COLOR_BGR2RGB)
        acm_img_2 = cv2.cvtColor(acm_img_2, cv2.COLOR_BGR2RGB)
        acm_img_1 = cv2.resize(acm_img_1,(H_1,W_1))
        acm_img_2 = cv2.resize(acm_img_2,(H_2,W_2))

        #元絵と合成
        mixed_img_1 = (np.float32(acm_img_1)*0.6 + img_1 *0.4)
        mixed_img_2 = (np.float32(acm_img_2)*0.6 + img_2 *0.4)

        #表示
        out_img_1 = np.concatenate((img_1, acm_img_1, mixed_img_1), axis=1)
        out_img_2 = np.concatenate((img_2, acm_img_2, mixed_img_2), axis=1)
        display_png(array_to_img(out_img_1))
        display_png(array_to_img(out_img_2))

# # 実際に予測してみる

# +
X_target_1 = []
X_target_2 = []
img = Image.open('image/photo/1.jpg')
img_face = Image.open('image/photo/face/1.jpg')
img = expand2square(img, (255, 255, 255))
img_face = expand2square(img_face, (255, 255, 255))
img = img.resize((img_size, img_size))
img_face = img_face.resize((img_size, img_size))
X_target_1.append(np.array(img))
X_target_2.append(np.array(img_face))
X_target_1 = np.array(X_target_1)
X_target_2 = np.array(X_target_2)

pred_1 = model_1.predict(X_target_1).flatten()
pred_2 = model_2.predict(X_target_2).flatten()
print('全体', pred_1)
print('顔', pred_2)
print('予測', (pred_1+pred_2)/2)

#df_pred_mean = (pred_1+pred_2)/2

#df_pred_predict = pd.DataFrame([pred_1, pred_2, df_pred_mean, y_test_1])
#df_pred_predict = df_pred_predict.T
#df_pred_predict.columns = ['全体', '顔', '平均', '実際の値']
#df_pred_predict.to_csv("pred_predict.csv", index=False)

#df_pred_predict

# +
   
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

# ## MSE   = Mean Squared Error        = 平均二乗誤差
# # $\frac{1}{n}\sum_{i=1}^{n}(f_i-y_i)^2$
#
# ## RMSE = Root Mean Square Error = 平均平方二乗誤差
# ルートの中身を二乗しているため、MAEと比較して外れ値をより大きな誤差として扱う傾向がある
# # $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(f_i-y_i)^2}$
#
# ## MAE   = Mean Absolute Error        = 平均絶対誤差
# # $\frac{1}{n}\sum_{i=1}^{n}|f_i-y_i|$
#
# #### RMSPE = Root Mean Square Percentage Error = 平均平方二乗誤差率
# ### $\sqrt{\frac{100}{n}\sum_{i=1}^{n}(\frac{f_i-y_i}{y_i})^2}$
#
# #### MAPE    = Mean Absolute Percentage Error = 平均絶対誤差率
# ### $\frac{100}{n}\sum_{i=1}^{n}|\frac{f_i-y_i}{y_i}|$
#
# |指標名  |特徴　                        |損失関数  |最適化指標|評価関数|KPI(重要標石評価指標)|
# |--------|------------------------------|----------|----------|--------|---------------------|
# |RMSE    |大きいエラーを重要視する      |◯        |◯        |△      |X                    |
# |RMSLE   |予測と実測の比率と下振れを重視|△        |◯        |△      |X                    |
# |MAE     |誤差の幅を等しく扱う          |△        |◯        |◯      |△                   |
# |MAPE    |誤差を百分率で考慮            |X         |△        |◯      |◯                   |
# |Poisson |離散確率分布                  |△(BP依存)|◯        |△      |X                    |
# |Gammma  |連続確率分布                  |△(BP依存)|◯        |△      |X                    |
# |Tweedie |0か0以外か                    |△(BP依存)|◯        |△      |X                    |
# |R Square|決定係数                      |X         |△        |△      |X                    |

# def create_model(model_type, img_width, img_height, img_channels):
#     model = Sequential()
#     
#     if model_type == 'CNN':
#         model.add(Conv2D(
#             filters = 16,
#             kernel_size = (2, 2),
#             strides = 1,
#             padding = 'same',
#             input_shape=(img_height, img_width, img_channels)))
#         model.add(PReLU())
#         model.add(Dropout(0.2))
#         
#         """
#         model.add(Conv2D(
#             filters = 16,
#             kernel_size = (1, 2),
#             strides = 1,
#             padding = 'same',
#             activation = 'relu',
#             input_shape=(img_height, img_width, img_channels)))
#         
#         model.add(Conv2D(
#             filters = 16,
#             kernel_size = (2, 1),
#             strides = 1,
#             padding = 'same',
#             activation = 'relu',
#             input_shape=(img_height, img_width, img_channels)))
#         """
#         
#         #model.add(BatchNormalization())
#         model.add(MaxPooling2D(pool_size=(2,2)))
#         ##model.add(Dropout(0.2))
#         
#         model.add(Conv2D(
#             filters = 32,
#             kernel_size = (2, 2),
#             strides = 1,
#             padding = 'same'))
#         model.add(PReLU())
#         #model.add(Dropout(0.2))
#         
#         """
#         #model.add(BatchNormalization())
#         model.add(MaxPooling2D(pool_size=(2,2)))
#         #model.add(Dropout(0.25))
#         
#         model.add(Conv2D(
#             filters = 64,
#             kernel_size = (3, 3),
#             strides = 1,
#             padding = 'same',
#             activation = 'relu'))
#         """
#         
#         #model.add(BatchNormalization())
#         model.add(Flatten())
#         model.add(Dense(32))
#         model.add(PReLU())
#         #model.add(Dropout(0.2))
#         model.add(Dense(1))
#         model.add(PReLU())
#         model.compile(
#             loss='mse',
#             optimizer='Adam',
#             metrics=['mae'])
#         
#     elif model_type == 'CNN2':    
#         # 畳み込み層
#         model.add(Conv2D(
#             filters = 16, # フィルターの枚数
#             kernel_size = (3, 3), # フィルターのサイズ
#             strides = 1, # フィルターの移動幅
#             padding='same', # 画像の周りを0パディングすることで端の特徴を捉えられるようにする
#             activation='relu', # 活性化関数
#             input_shape=(img_height, img_width, img_channels))) # 画像のサイズ
#         
#         # 畳み込み層
#         model.add(Conv2D(
#             filters = 16, # フィルターの枚数
#             kernel_size = (3, 3), # フィルターのサイズ
#             strides = 1, # フィルターの移動幅
#             padding='same', # 画像の周りを0パディングすることで端の特徴を捉えられるようにする
#             activation='relu', # 活性化関数
#             input_shape=(img_height, img_width, img_channels))) # 画像のサイズ
#     
#         # プーリング層（情報量の削減）
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         
#         # ドロップアウト層
#         model.add(Dropout(0.25))
#         
#         #"""
#         # 畳み込み層
#         model.add(Conv2D(
#             filters = 32,
#             kernel_size = (3, 3),
#             strides = 1,
#             padding='same',
#             activation='relu'))
#         
#         # 畳み込み層
#         model.add(Conv2D(
#             filters = 32,
#             kernel_size = (3, 3),
#             strides = 1,
#             padding='same',
#             activation='relu'))
#         
#         # プーリング層
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         
#         # ドロップアウト層
#         model.add(Dropout(0.25))
#         #"""
#         
#         # 多層パーセプトロンに入力するにあたり４次元配列を１次元配列に変換
#         model.add(Flatten())
#         
#         # 全結合層
#         #model.add(Dense(512, activation='relu'))
#         
#         # ドロップアウト層
#         #model.add(Dropout(0.5))
#         
#         # 全結合層
#         model.add(Dense(1, activation='relu'))
#         
#         # 活性化層
#         #model.add(Activation('softmax'))
#         #model.add(Activation('relu'))
#         
#         # コンパイル
#         model.compile(
#             loss='mse',
#             optimizer='Adam',
#             metrics=['mae'])
#         
#     elif model_type == 'MyVGG16':
#         # 畳み込み層
#         model.add(Conv2D(
#             32, (3, 3), # フィルタ数とフィルタサイズ
#             padding='same', # 0パディング
#             activation='relu', # 活性化関数
#             input_shape=(img_height, img_width, img_channels))) # 画像のサイズ
#         
#         # 畳み込み層
#         model.add(Conv2D(
#             32, (3, 3), # 16は？。(3, 3)はフィルタのサイズ
#             padding='same', # ?
#             activation='relu', # 活性化関数
#             input_shape=(img_height, img_width, img_channels))) # 画像のサイズ
#         
#         # 畳み込み層
#         model.add(Conv2D(
#             32, (3, 3), # 16は？。(3, 3)はフィルタのサイズ
#             padding='same', # ?
#             activation='relu', # 活性化関数
#             input_shape=(img_height, img_width, img_channels))) # 画像のサイズ
#         
#         # 畳み込み層
#         #model.add(Conv2D(
#         #    16, (3, 3), # 16は？。(3, 3)はフィルタのサイズ
#         #    padding='same', # ?
#         #    activation='relu', # 活性化関数
#         #    input_shape=(img_height, img_width, img_channels))) # 画像のサイズ
#     
#         # プーリング層（情報量の削減）基本は(2, 2)
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         
#         # ドロップアウト層
#         model.add(Dropout(0.25))
#         
#         
#         # 畳み込み層
#         model.add(Conv2D(
#             32, (3, 3),
#             padding='same',
#             activation='relu'))
#         
#         # 畳み込み層
#         model.add(Conv2D(
#             32, (3, 3),
#             padding='same',
#             activation='relu'))
#         
#         # 畳み込み層
#         model.add(Conv2D(
#             32, (3, 3),
#             padding='same',
#             activation='relu'))
#         
#         # 畳み込み層
#         #model.add(Conv2D(
#         #    32, (3, 3),
#         #    padding='same',
#         #    activation='relu'))
#         
#         # プーリング層
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         
#         # ドロップアウト層
#         model.add(Dropout(0.25))
#         
#         # 多層パーセプトロンに入力するにあたり４次元配列を１次元配列に変換
#         model.add(Flatten())
#         
#         # 全結合層
#         model.add(Dense(512, activation='relu'))
#         
#         # ドロップアウト層
#         model.add(Dropout(0.5))
#         
#         # 全結合層
#         model.add(Dense(1))
#         
#         # 活性化層
#         #model.add(Activation('softmax'))
#         model.add(Activation('relu'))
#         
#         # コンパイル
#         model.compile(
#             loss='mse',
#             optimizer='Adam',
#             metrics=['mae'])
#         
#     elif model_type == 'VGG16':
#         # VGG16
#         model = VGG16(
#             include_top=True,
#             weights=None,
#             input_tensor=None,
#             input_shape=(img_width, img_height, img_channels),
#             #pooling=None,
#             classes=1,
#         )
#         
#         
#         # 全結合層
#         #model.add(Dense(NUM_CLASSES))
#         
#         # 活性化層
#         #model.add(Activation('softmax'))
#         
#         # コンパイル
#         model.compile(
#             loss='mse',
#             optimizer='Adam',
#             metrics=['mae'])
#         
#     elif model_type == 'VGG19':
#         model = VGG19(
#             include_top=True,
#             weights=None,
#             input_tensor=None,
#             input_shape=(img_width, img_height, img_channels),
#             #pooling=None,
#             classes=1
#         )
#         
#         
#         # コンパイル
#         model.compile(
#             loss='mse',
#             optimizer='Adam',
#             metrics=['mae'])
#         
#     elif model_type == 'ResNet50':
#         model = ResNet50(
#             include_top=True,
#             weights=None,
#             input_tensor=None,
#             input_shape=(img_width, img_height, img_channels),
#             #pooling=None,
#             classes=1
#         )
#         
#         # コンパイル
#         model.compile(
#             loss='mse',
#             optimizer='Adam',
#             metrics=['mae'])
#         
#     elif model_type == 'InceptionResNetV2':
#         model = InceptionResNetV2(
#             include_top=True,
#             weights=None,
#             input_tensor=None,
#             input_shape=(img_width, img_height, img_channels),
#             #pooling=None,
#             classes=2
#         )
#         
#         # コンパイル
#         model.compile(
#             loss='mse',
#             optimizer='Adam',
#             metrics=['mae'])
#         
#     return model


