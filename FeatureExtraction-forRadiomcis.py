import logging

import SimpleITK as sitk

import radiomics
import radiomics.featureextractor as FEE

import os,re,csv,xlrd
import numpy as np
import pandas as pd

import time 

def GetFeature(imageName,maskName,para_path):
#    print("originl path: " + ori_path)
#    print("label path: " + lab_path)
#    print("parameter path: " + para_path)
    
    if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
      print('Error getting testcase!')
      exit()
    
    # Get the PyRadiomics logger (default log-level = INFO)
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file
    # Set up the handler to write out all log entries to a file
    handler = logging.FileHandler(filename='testLog.txt', mode='w')
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # 使用配置文件初始化特征抽取器
    extractor = FEE.RadiomicsFeaturesExtractor(para_path)
#    print ("Extraction parameters:\n\t", extractor.settings)
#    print ("Enabled filters:\n\t", extractor._enabledImagetypes)
#    print ("Enabled features:\n\t", extractor._enabledFeatures)
    
    # 运行
    result = extractor.execute(imageName,maskName)  #抽取特征
#    print ("Result type:", type(result))  # result is returned in a Python ordered dictionary
#    print ("Calculated features")
#    for key, value in result.items():  #输出特征
#        print ("\t", key, ":", value)
    return result

def saveFeature_as_list(mydict):
    csvlist = []
    for key, value in mydict.items():  #输出特征
#        print ("\t", key, ":", value)
        csvlist.append([key,str(value)])  
    return csvlist
#-----------------------------------------------------------------------------#
def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
 
    isExists=os.path.exists(path) 
    if not isExists:
        os.makedirs(path)  
        print ('create folder:' +path+' complete!')
        return True
    else:
        print (path+' is exist! Will not create folder')
        return False

def load_xlsx(path,sheetid):
    data = xlrd.open_workbook(path)
    table = data.sheets()[sheetid] # 0,1,2...
    nrows = table.nrows
    info = []
    for i in range(nrows):
        info.append(table.row_values(i)[:])
    return info

#-----------------------------------------------------------------------------#
        
main_path = 'I:\PCa\BCR_new\data\pyradiomics_features'
hosp =  "CC"
seq = 'T2'
ids = sorted(next(os.walk(os.path.join(main_path,hosp)))[1])

L=[]
L_time = []
for i in range(len(ids)):
    start = time.time()
    idx=ids[i]
    srcPath=os.path.join(main_path,hosp,idx,seq+'.csv')
    df = pd.read_csv(srcPath) 
    L.append([idx]+list(df['1'])[21:])
    end = time.time()
    L_time.append(end-start)
print('Runtime| total={}, per_patient{}±{}'.format(np.sum(L_time),np.mean(L_time),np.std(L_time)))
df_save = pd.DataFrame(L,columns=['ID']+list(df['0'])[21:])
df_save.to_csv(os.path.join(main_path,hosp+'-ft.csv'),index=0)        
        
        
        
        
        
        
        