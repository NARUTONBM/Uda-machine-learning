import os
import numpy as np
import random
from PIL import Image
from keras.utils import np_utils
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

TRAIN_DIR = "data/train/"
TEST_DIR = "data/test/"
rows = 224
cols = 224
channels = 3

# 获取图片路径
train_images = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR)]
random.shuffle(train_images)
train_dogs = [i for i in train_images if 'dog' in i]
train_cats = [i for i in train_images if 'cat' in i]
test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]


def path_to_tensor(path):
    # PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(path, target_size=(rows, cols))
    # 转成3维张量
    x = image.img_to_array(img)
    # 转成（1，224，224，3的4维张量返回
    return np.expand_dims(x, axis=0)


def paths_to_tensor(paths):
    list_of_tensor = [path_to_tensor(path) for path in paths]
    return np.vstack(list_of_tensor)


train_tensors = paths_to_tensor(train_images).astype('float32') / 255


# test_tensors = paths_to_tensor(test_images).astype('float32') / 255


def get_targets(paths):
    targets = []

    for i in paths:
        if 'dog' in i:
            targets.append(1)
        else:
            targets.append(0)

    targets = np_utils.to_categorical(targets, 133)
    return targets


train_targets = get_targets(train_images)
test_targets = get_targets(test_images)

# 从Xception预训练的CNN获取bottleneck特征
bottleneck_features = np.load('data/bottleneck_features/DogXceptionData.npz')
train_Xception = bottleneck_features['train']
valid_Xception = bottleneck_features['valid']
test_Xception = bottleneck_features['test']

Xception_model = Sequential()
# 定义网络结构
Xception_model.add(GlobalAveragePooling2D(input_shape=train_Xception.shape[1:]))
Xception_model.add(Dense(133, activation='softmax'))

Xception_model.summary()
# 编译模型
Xception_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
check_pointer = ModelCheckpoint(filepath='saved_models/weights.best.Xception.hdf5', verbose=1, save_best_only=True)

Xception_model.fit(train_tensors, train_targets, validation_data=0.33, epochs=20, batch_size=20,
                   callbacks=[check_pointer], verbose=1)
# 加载具有最佳验证loss的模型权重
Xception_model.load_weights('saved_models/weights.best.Xception.hdf5')

# 在测试集上计算分类准确率
Xception_predictions = [np.argmax(Xception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Xception]

test_accuracy = 100 * np.sum(np.array(Xception_predictions) == np.argmax(test_targets, axis=1)) / len(Xception_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
