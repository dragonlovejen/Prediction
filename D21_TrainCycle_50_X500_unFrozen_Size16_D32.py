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

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.utils import class_weight
import math
###
def ExtendData(pd_Data_Ori,curSeed_):
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

    ###
    pd_Data_1 = pd_Data_Ori.copy()
    pd_Data_1.reset_index(inplace=True,drop = True)
    pd_Data_1 = ChoiceToZero(pd_Data_1)

    pd_Data_2 = pd_Data_Ori.copy()
    pd_Data_2.reset_index(inplace=True,drop = True)
    pd_Data_2 = ChoiceToZero(pd_Data_2)

    pd_Data_3 = pd_Data_Ori.copy()
    pd_Data_3.reset_index(inplace=True,drop = True)
    pd_Data_3 = ChoiceToZero(pd_Data_3)

    #####
    pd_Data_4 = pd_Data_Ori.copy()
    pd_Data_4.reset_index(inplace=True,drop = True)
    pd_Data_4 = ChoiceToZero(pd_Data_4)

    pd_Data_5 = pd_Data_Ori.copy()
    pd_Data_5.reset_index(inplace=True,drop = True)
    pd_Data_5 = ChoiceToZero(pd_Data_5)
    #####

    pd_Data_6 = pd_Data_Ori.copy()
    pd_Data_6.reset_index(inplace=True,drop = True)
    pd_Data_6 = ChoiceToZero(pd_Data_6)

    pd_Data_7 = pd_Data_Ori.copy()
    pd_Data_7.reset_index(inplace=True,drop = True)
    pd_Data_7 = ChoiceToZero(pd_Data_7)

    pd_Data_8 = pd_Data_Ori.copy()
    pd_Data_8.reset_index(inplace=True,drop = True)
    pd_Data_8 = ChoiceToZero(pd_Data_8)

    pd_Data_9 = pd_Data_Ori.copy()
    pd_Data_9.reset_index(inplace=True,drop = True)
    pd_Data_9 = ChoiceToZero(pd_Data_9)

    pd_Data_10 = pd_Data_Ori.copy()
    pd_Data_10.reset_index(inplace=True,drop = True)
    pd_Data_10 = ChoiceToZero(pd_Data_10)
    #####


    pd_Extend = pd.concat([pd_Data_Ori,pd_Data_1,pd_Data_2,pd_Data_3,pd_Data_4,pd_Data_5,
                           pd_Data_6,pd_Data_7,pd_Data_8,pd_Data_9,pd_Data_10,] ,
                        axis = 0 , ignore_index = True)


    pd_Extend = shuffle(pd_Extend,random_state = curSeed_ )

    return pd_Extend
######

myCycle = 42
for myI in range(0,myCycle,1):
    print('Cycle = ',myI ,  '-'*50)

    EraseRate = 0.2
    myTimes   = 10
    myEpochs  = 6
    myBatch   = 16
    ###
    ofile_path = r'E:\RunResult\\'
    ofile_name = 'D21_PreUnFrozen_Bath16_32_X500_Tim_'



    preTrain_model = r'E:\D01_PreTrain_50_E8_X500_D32.h5'

    sfile = r'E:\B21_StrCut_ToArray_ForTarget_X500.xlsx'

    pd_Data = pd.read_excel(sfile , sheet_name = 'pd_Data' )
    popCols = ['Link01','MainStr_1','X000' ,'StrCut']

    for col in popCols:
        pd_Data.pop(col)
    #####

    pd_null = pd.DataFrame({'...' : [None]  })
    Kscores = []
    for i in range(1,myTimes+1,1):

        ##
        curSeed = np.random.randint(100000000, size=1)
        np.random.seed(curSeed)
        ##

        pd_Data = shuffle(pd_Data ,random_state = curSeed[0])


        data_num = pd_Data.shape[0]
        train_data_Ori = pd_Data[:int(data_num *0.7)].copy()
        test_data  = pd_Data[int(data_num *0.7):].copy()
        train_data = ExtendData(train_data_Ori , curSeed[0])

        ###
        class_weights = class_weight.compute_class_weight('balanced',
                                                    classes = [0 , 1 , 2],
                                                    y = train_data['WIN_Remark'])

        weights = { i : round(w,6) for i , w in enumerate(class_weights)}



        ########## train
        x_np_train_data  = pd.DataFrame(train_data.drop('WIN_Remark', axis='columns'))

        train_data['WIN_Remark'] = train_data['WIN_Remark'].astype('category')
        y_train_label = to_categorical(train_data['WIN_Remark'])

        #### test
        x_np_test_data  = pd.DataFrame(test_data.drop('WIN_Remark', axis='columns'))
        test_data['WIN_Remark'] = test_data['WIN_Remark'].astype('category')
        y_test_label = to_categorical(test_data['WIN_Remark'])

        ########
        model = None
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
        model.load_weights(preTrain_model, by_name=True )
        # model.layers[0].trainable = False   #unFrozen the Embedding layer of Pre Train Model

        model.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['acc'])
        ########

        ###
        # history = model.fit(x_np_train_data, y_train_label,
        #                     epochs=myEpochs,
        #                     batch_size=myBatch,
        #                     class_weight=class_weight,
        #                     validation_split=0.05, verbose=0 )

        history = model.fit(x_np_train_data, y_train_label,
                            epochs=myEpochs,
                            batch_size=myBatch,
                            class_weight=weights,
                            verbose=0 )

        # model.save(outfile_model)

    # ##########################
    #     ###########
    #     history_dict = history.history
    #     # print(history_dict.keys())
    #     loss_values = history_dict['loss']
    #     epochs = range(1, len(loss_values)+ 1)
    #     ###########
    #     acc = history_dict['acc']
    #     val_acc = history_dict['val_acc']
    #     epochs = range(1, len(loss_values)+ 1)
    #     plt.plot(epochs, acc, 'b', label='Training acc')
    #     plt.plot(epochs, val_acc, 'g', label='Validation acc')
    #     plt.plot(epochs, loss_values, 'r', label='loss values')
    #     plt.title('Training and validation accuracy')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Accuracy')
    #     plt.legend()
    #
    #     plt.show()
    #     ###########
    # ##########################

        ###
        scores = model.evaluate(x_np_test_data, y_test_label, verbose=0 )
        print('myTimes = ' , i ,' curSeed = ' , curSeed , '*'*30 )

        print('i = ' , i , " %s: %.1f%%" % (model.metrics_names[1], scores[1]*100))
        Kscores.append(scores[1] * 100)

        #######
        y_pred_ = model.predict(x_np_test_data)
        y_predict = np.argmax(y_pred_, axis=1)

        pd_test_predict = pd.DataFrame({'y_Predict' : y_predict})
        y_act = np.argmax(y_test_label, axis=1)
        pd_test_predict['WIN_Remark'] = y_act


        locals()['Note_'+str(i)] = pd.DataFrame({'curSeed' : [curSeed] ,
                                                'class_weight' : [weights] ,
                                                model.metrics_names[1] : [round(scores[1],3)*100]})

        locals()['crosstab_'+str(i)] = pd.crosstab(pd_test_predict['WIN_Remark'], pd_test_predict['y_Predict'], rownames=['WIN_Remark'], colnames=['Predicted'])

        pd_report = pd.DataFrame(classification_report(y_act, y_predict, output_dict=True))
        pd_report = pd_report.round(4)
        locals()['classification_report_'+str(i)] = pd_report

        locals()['Report_'+str(i).zfill(2)] = pd.concat([locals()['crosstab_'+str(i)] ,
                                            pd_null ,
                                            locals()['classification_report_'+str(i)],
                                            pd_null ,
                                            locals()['Note_'+str(i)] ] , axis = 1)

        #######

        ###########

    print('*'*80)

    print("%.1f%% (+/- %.1f%%)" % (np.mean(Kscores), np.std(Kscores)))

    myMean = str(int(round(np.mean(Kscores),3)*100))
    ofile = ofile_path  + ofile_name + str(myTimes) + '_8Epo'  + str(myEpochs) + '_' + myMean  + '.xlsx'

    with pd.ExcelWriter(ofile) as writer :
        for i in range(1,myTimes+1,1):
            locals()['Report_'+str(i).zfill(2)].to_excel(writer , index = True , sheet_name = 'Report_'+str(i).zfill(2) , freeze_panes=[1,1])
