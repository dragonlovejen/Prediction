import numpy as np
import pandas as pd
import pyodbc
import sqlite3
from datetime import date, datetime, timedelta
import time
import os


Source_Xlsx = r'E:\CreateData_forPydb'
Pydb_LV1 = r'E:\Pydb_LV1.db'


for name_yyy in os.listdir(Source_Xlsx):
    subPath_yyy = os.path.join(Source_Xlsx,name_yyy)
    myI = 0

    for name_mmm in os.listdir(subPath_yyy):
        subPath_mmm = os.path.join(subPath_yyy,name_mmm)
        if myI == 0 :
            pd_ALL = pd.read_excel(subPath_mmm ,sheet_name = 'DATA',dtype = str)
        else:
            pd_Temp = pd.read_excel(subPath_mmm ,sheet_name = 'DATA',dtype = str)
            pd_ALL  = pd.concat([pd_ALL , pd_Temp] , sort=False)
        myI += 1

    ###
    # DD.JLOC || DD.JYEAR || DD.JCASE || '_' || DD.JNO as Link01
    pd_ALL['Link01'] = pd_ALL['JLOC'] + pd_ALL['JYEAR'] + pd_ALL['JCASE'] +  '_' + pd_ALL['JNO']
    # print(pd_ALL.head())
    ###

    #####
    myPydb_LV1 = sqlite3.connect(Pydb_LV1)
    curPy_LV1  = myPydb_LV1.cursor()
    pd_ALL.to_sql('DATA_LV1',myPydb_LV1 ,if_exists='append',index=False)
    time.sleep(1.5)
    curPy_LV1.close()
    #####


