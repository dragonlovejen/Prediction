import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from tensorflow.keras import optimizers
from sklearn.utils import shuffle
import math
###
def ExtendData(pd_Data_Ori):
    def ChoiceToZero(pd_Inner):

        for i_Inner , d in enumerate(pd_Inner['X001']):
            OriData = pd_Inner.iloc[i_Inner,1:]
            noZero_Len = np.count_nonzero(OriData)
            colNames = ['X' + str(i_i +1).zfill(3) for i_i in range(noZero_Len)]
            cho_Size = math.ceil(noZero_Len * EraseRate)
            cho_Cols = np.random.choice(colNames , size = cho_Size)
            for cc in cho_Cols:
                pd_Inner.loc[i_Inner , cc] = 0
        return pd_Inner


    pd_Data_1 = pd_Data_Ori.copy()
    pd_Data_1.reset_index(inplace=True,drop = True)
    pd_Data_1 = ChoiceToZero(pd_Data_1)

    pd_Data_2 = pd_Data_Ori.copy()
    pd_Data_2.reset_index(inplace=True,drop = True)
    pd_Data_2 = ChoiceToZero(pd_Data_2)

    pd_Data_3 = pd_Data_Ori.copy()
    pd_Data_3.reset_index(inplace=True,drop = True)
    pd_Data_3 = ChoiceToZero(pd_Data_3)

    pd_Data_4 = pd_Data_Ori.copy()
    pd_Data_4.reset_index(inplace=True,drop = True)
    pd_Data_4 = ChoiceToZero(pd_Data_4)

    pd_Data_5 = pd_Data_Ori.copy()
    pd_Data_5.reset_index(inplace=True,drop = True)
    pd_Data_5 = ChoiceToZero(pd_Data_5)

    pd_Extend = pd.concat([pd_Data_Ori,pd_Data_1,pd_Data_2,pd_Data_3,pd_Data_4,pd_Data_5] ,
                        axis = 0 , ignore_index = True)


    pd_Extend = shuffle(pd_Extend)

    return pd_Extend

EraseRate = 0.2
myEpochs = 8
myBatch = 2048

sfile = r'E:\B21_StrCut_ToArray_ForPreModel_X500.xlsx'

outfile_model = r'E:\D01_PreTrain_50_E8_X500_D32.h5'

pd_Data = pd.read_excel(sfile , sheet_name = 'pd_Data')

popCols = ['Link01','MainStr_1','X000' ,'StrCut']
for col in popCols:
    pd_Data.pop(col)

#####
data_num = pd_Data.shape[0]
indexes  = np.random.permutation(data_num)

train_indexes = indexes[:int(data_num *0.99)]
valid_indexes = indexes[int(data_num *0.99):]
train_data_Ori  = pd_Data.loc[train_indexes]
valid_data  = pd_Data.loc[valid_indexes]


train_data = ExtendData(train_data_Ori)

print(train_data.shape,valid_data.shape)


##########
x_np_train_data  = pd.DataFrame(train_data.drop('WIN_Remark', axis='columns'))
train_data['WIN_Remark'] = train_data['WIN_Remark'].astype('category')
y_train_label = to_categorical(train_data['WIN_Remark'])

####
x_np_valid_data  = pd.DataFrame(valid_data.drop('WIN_Remark', axis='columns'))
valid_data['WIN_Remark'] = valid_data['WIN_Remark'].astype('category')
y_valid_label = to_categorical(valid_data['WIN_Remark'])


######## Ori
model = models.Sequential()
model.add(layers.Embedding(1000000, 50, input_length=500,name='word2vec'))

model.add(layers.Conv1D(filters=256,
                    kernel_size=3,
                    padding='valid',
                    activation='relu',
                    strides=1))

model.add(layers.MaxPooling1D(pool_size=3))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(32, activation='relu'))
model.add(layers.Flatten())

model.add(layers.Dense(3, activation='softmax'))
######## Ori


model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['acc'])

###
# history = model.fit(x_np_train_data, y_train_label,
#                     epochs=myEpochs,
#                     batch_size=myBatch,
#                     validation_split=0.2)

# history = model.fit(x_np_train_data, y_train_label,
#                     epochs=myEpochs,
#                     batch_size=myBatch,
#                     class_weight=class_weight,
#                     validation_data=(x_np_valid_data, y_valid_label))

history = model.fit(x_np_train_data, y_train_label,
                    epochs=myEpochs,
                    batch_size=myBatch,
                    validation_data=(x_np_valid_data, y_valid_label))

model.save(outfile_model)

# ###########
# history_dict = history.history
# print('*'*80)
# print(history_dict.keys())
# loss_values = history_dict['loss']
# # val_loss_values = history_dict['val_loss']
# epochs = range(1, len(loss_values)+ 1)
#
# ###########
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']
# # print(acc)
# # print('*'*80)
# # print(val_acc)
#
# epochs = range(1, len(loss_values)+ 1)
#
# plt.plot(epochs, acc, 'b', label='Training acc')
# plt.plot(epochs, val_acc, 'g', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.show()
# ###########
