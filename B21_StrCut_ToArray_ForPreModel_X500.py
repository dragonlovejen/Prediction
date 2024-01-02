
import jieba
import numpy as np
import pandas as pd

sfile_zhWords = r'E:\dict_txt_big_jieba_繁體中文.txt'
sfile_zhXlsx = r'E:\dict_txt_big_jieba_繁體中文_Single.xlsx'

pd_Words = pd.read_excel(sfile_zhXlsx , sheet_name = 'pd_zhStr')
dict_Words = {word : strCode for word , strCode in zip(pd_Words['Words'],pd_Words['StrCode'])}



sfile_Data = r'E:\A51_GetStringAfterWinRemark_fPydb.xlsx'

ofile = r'E:\B21_StrCut_ToArray_ForPreModel_X500.xlsx'

# ## Name for Ori data : pd_TTL
# pd_Data_Ori = pd.read_excel(sfile_Data , sheet_name = 'pd_TTL')

# ## Name for pre tain data : pd_PreModelData
pd_Data_Ori = pd.read_excel(sfile_Data , sheet_name = 'pd_noTarget')

###


### For Pre train Model Data
wanCols = ['Link01' ,'WIN_Remark','MainStr_1']
pd_Data = pd_Data_Ori[wanCols].copy()
#######
col_Len = 501
colNames = ['X' +str(i).zfill(3) for i in range(col_Len)]
for i , col in enumerate(colNames):
    pd_Data[col] = 0
###

jieba.set_dictionary(sfile_zhWords)
StrCut = []
# StrCut
X000_List = []
for i , myStrs in enumerate(pd_Data['MainStr_1']):
    try:
        myCut = jieba.lcut(myStrs, cut_all=False)
        _myCut = '|'.join(myCut)
        StrCut.append(_myCut)
        X000_List.append('V')
        ###
        if len(myCut) >0 :
            myC = 4
            for ii , Word in enumerate(myCut):
                try:
                    pd_Data.iloc[i,myC] = dict_Words[Word]
                    myC += 1
                except:
                    pass

    except:
        StrCut.append('')
        X000_List.append('X')


pd_Data['StrCut'] = StrCut
pd_Data['X000'] = X000_List

flg_Yes = (pd_Data['X000'] == 'V')
pd_Data = pd_Data.loc[flg_Yes,:]

with pd.ExcelWriter(ofile) as writer :
    pd_Data.to_excel(writer , index = False , sheet_name = 'pd_Data' , freeze_panes=[1,1])
