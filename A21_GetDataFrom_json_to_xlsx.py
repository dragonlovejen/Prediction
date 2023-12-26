import sys, win32com.client
import os
import json
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

from datetime import datetime
import time


def GetAndSwitchJson(sourPath ,ofile):
    myKeeps = ',訴,'

    myLOC , myYEAR, myCASE  = [],[],[]
    myDATE_Ori , myDATE = [],[]
    myNO  , myCLASS = [],[]

    myAdjType = []
    ###


    mySuR_01 , mySuR_02 , mySuR_03 = [],[],[]

    mySuWL_b1 , mySuWL_b2 = [] ,[]
    mySuWL_01 , mySuWL_02 , mySuWL_03 = [],[],[]
    mySuWL_04 , mySuWL_05 , mySuWL_06 = [],[],[]
    mySuWL_07  = []

    mySuWL_44_temp , mySuWL_66_temp = [],[]

    strSuWL_b1 = '駁回'
    strSuWL_b2 = '被告應'

    strSuWL_01 = '原告之訴駁回'
    strSuWL_02 = '原告其餘之訴駁回'

    strSuWL_03 = '訴訟費用由兩造'

    strSuWL_04 = '訴訟費用由原告'
    strSuWL_05 = '訴訟費用由原告__分之__'

    strSuWL_06 = '訴訟費用由被告'
    strSuWL_07 = '訴訟費用由被告__分之__'


    #######
    outSoup_1 = "<html><body><p>"
    outSoup_2 = "</p></body></html>"
    #######

    # myJFs_ori      = []
    myJFs_outSpace = [] # out space
    myJFs_MainStr  = [] # MainStr

    for name in os.listdir(sourPath):
        ########
        myStrs = []
        if myKeeps in name  :
            myStrs = name[:-5].split(',')
            try:
                myLOC.append(myStrs[0])
            except:
                myLOC.append('')

            try:
                myYEAR.append(myStrs[1])
            except:
                myYEAR.append('')

            try:
                myCASE.append(myStrs[2])
            except:
                myCASE.append('')

            try:
                myNO.append(myStrs[3])
            except:
                myNO.append('')

            try:
                myDATE_Ori.append(myStrs[4])
            except:
                myDATE_Ori.append('')

            try:
                myCLASS.append(myStrs[5])
            except:
                myCLASS.append('')
            ###
            _year = int(myStrs[4][:4])
            _mon  = int(myStrs[4][4:6])
            _day  = int(myStrs[4][6:])
            _date = datetime(_year , _mon , _day)
            myDATE.append(_date)
            ###

            fullName = os.path.join(sourPath,name)
            with open(fullName, 'r', encoding="utf-8") as inFile:
                json_Ori = json.load(inFile)
                soup_JF  = BeautifulSoup(json_Ori['JFULL'] ,"lxml")
                soupTxt_ori  = soup_JF.text.replace(outSoup_1,'').replace(outSoup_2,'')
                soupTxt_outSpace = soupTxt_ori.replace('\r\n','').replace('　','').replace(' ','')
                MainStrLoc = soupTxt_outSpace.find('主文')
                MainStrLoc_End = soupTxt_outSpace.find('理由')

                if MainStrLoc_End == -1 :
                    MainStrLoc_End = MainStrLoc + 500

                ### WIN/LOSE/HALF key words
                # strSuWL_b1 = '駁回'
                if strSuWL_b1   in soupTxt_outSpace[MainStrLoc:MainStrLoc+18] :
                    mySuWL_b1.append('1')
                else:
                    mySuWL_b1.append('0')

                # strSuWL_b2 = '被告應'
                if strSuWL_b2   in soupTxt_outSpace[MainStrLoc:MainStrLoc+8] :
                    mySuWL_b2.append('1')
                else:
                    mySuWL_b2.append('0')

                # strSuWL_01 = '原告之訴駁回'
                if strSuWL_01   in soupTxt_outSpace[MainStrLoc+18:MainStrLoc_End] :
                    mySuWL_01.append('1')
                else:
                    mySuWL_01.append('0')

                # strSuWL_02 = '原告其餘之訴駁回'
                if strSuWL_02  in soupTxt_outSpace[MainStrLoc+18:MainStrLoc_End] :
                    mySuWL_02.append('1')
                else:
                    mySuWL_02.append('0')

                # strSuWL_03 = '訴訟費用由兩造'
                if strSuWL_03  in soupTxt_outSpace[MainStrLoc:MainStrLoc_End] :
                    mySuWL_03.append('1')
                else:
                    mySuWL_03.append('0')

                ###
                # strSuWL_04 = '訴訟費用由原告'
                # strSuWL_05 = '訴訟費用由原告__分之__'
                if strSuWL_04  in soupTxt_outSpace[MainStrLoc:MainStrLoc_End] :
                    LocBegin_04 = soupTxt_outSpace.find('訴訟費用由原告')
                    _str_44 = soupTxt_outSpace[LocBegin_04 : LocBegin_04 + 15]
                    mySuWL_44_temp.append(_str_44)

                    if '分' in _str_44 :
                        mySuWL_04.append('0')
                        mySuWL_05.append('1')
                    else:
                        mySuWL_04.append('1')
                        mySuWL_05.append('0')

                else:
                    mySuWL_44_temp.append('')
                    mySuWL_04.append('0')
                    mySuWL_05.append('0')

                ###

                ###
                # strSuWL_06 = '訴訟費用由被告'
                # strSuWL_07 = '訴訟費用由被告__分之__'
                if strSuWL_06  in soupTxt_outSpace[MainStrLoc:MainStrLoc_End] :
                    LocBegin_06 = soupTxt_outSpace.find('訴訟費用由被告')
                    _str_66 = soupTxt_outSpace[LocBegin_06 : LocBegin_06 + 15]
                    mySuWL_66_temp.append(_str_66)

                    if '分' in _str_66 :
                        mySuWL_06.append('0')
                        mySuWL_07.append('1')
                    else:
                        mySuWL_06.append('1')
                        mySuWL_07.append('0')

                else:
                    mySuWL_66_temp.append('')
                    mySuWL_06.append('0')
                    mySuWL_07.append('0')
                ###

                #######
                cur_Adj = soupTxt_ori[10:12]
                myAdjType.append(cur_Adj)
                myJFs_outSpace.append(soupTxt_outSpace)
                myJFs_MainStr.append(soupTxt_outSpace[MainStrLoc+2:MainStrLoc_End])
                ##


    ############
    if len(myLOC) > 0 : #有資料才輸出

        pd_Data_Ori = pd.DataFrame({
                'JLOC'     : myLOC ,
                'JYEAR'    : myYEAR ,
                'JCASE'    : myCASE ,
                'JNO'      : myNO ,
                'JDATE_Ori'  : myDATE_Ori ,
                'JDATE'    : myDATE ,

                'JCLASS'   : myCLASS ,
                'AdjType'  : myAdjType ,

                '駁回_inBegin'    : mySuWL_b1  ,
                '被告應_inBegin'  : mySuWL_b2  ,

                strSuWL_01  : mySuWL_01  ,
                strSuWL_02  : mySuWL_02  ,
                strSuWL_03  : mySuWL_03  ,

                '訴訟費用由原告_Strings' : mySuWL_44_temp ,
                strSuWL_04  : mySuWL_04  ,
                strSuWL_05  : mySuWL_05  ,

                '訴訟費用由被告_Strings' : mySuWL_66_temp ,
                strSuWL_06  : mySuWL_06  ,
                strSuWL_07  : mySuWL_07  ,

                'StrList'  : '' ,

                'MainStr'  : myJFs_MainStr ,
                'JF_outSpace'    : myJFs_outSpace ,
                })


        pd_Data_Ori['StrList'] =   pd_Data_Ori['駁回_inBegin'] + pd_Data_Ori['被告應_inBegin'] + pd_Data_Ori[strSuWL_01] + pd_Data_Ori[strSuWL_02] + pd_Data_Ori[strSuWL_03] + pd_Data_Ori[strSuWL_04]+ pd_Data_Ori[strSuWL_05] + pd_Data_Ori[strSuWL_06] + pd_Data_Ori[strSuWL_07]


        with pd.ExcelWriter(ofile) as writer:
            pd_Data_Ori.to_excel(writer ,index = False , sheet_name='DATA',freeze_panes=[1,2])



dataPath = r'E:\Dragon_論文Data'
outPath  = r'E:\Dragon_論文\PyCode_V061\CreateData_forPydb'

# dataPath = r'E:\Dragon_論文Data_ForShortTry'
# outPath  = r'E:\Dragon_論文\PyCode_V03\CreateData_forPydb_ForShortTry'


##### Notes
# # dataPath\name_yyy   \name_mmm   \name_loc   \json_name_file
# # dataPath\subPath_yyy\subPath_mmm\subPath_loc\
# # outPath\outPath_yyy
##### Notes

for name_yyy in os.listdir(dataPath):
    subPath_yyy = os.path.join(dataPath,name_yyy)

    ###
    outPath_yyy = os.path.join(outPath,name_yyy)
    try:
        os.mkdir(outPath_yyy)
    except:
        pass
    ###

    for name_mmm in os.listdir(subPath_yyy):
        subPath_mmm = os.path.join(subPath_yyy,name_mmm)
        for name_loc in os.listdir(subPath_mmm):

            if '民事' in name_loc  :
                cur_ofile = outPath_yyy  + '\\' + name_loc + '_' + name_mmm + '.xlsx'
                subPath_loc = os.path.join(subPath_mmm,name_loc)
                GetAndSwitchJson(subPath_loc,cur_ofile)


