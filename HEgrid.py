import openslide

import os,re,xlrd,imageio,cv2,sys
import numpy as np
import matplotlib.pyplot as plt

from xml.dom.minidom import parse
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

#-----------------------------------------------------------------------------#
def fromXMLGetMask(slide_xml,slidesize,rate,regionId='1'):
    canvas = np.zeros([slidesize[1],slidesize[0]],dtype=np.uint8)
    
    doc = parse(slide_xml)
    element_obj  = doc.documentElement
    Annotation = element_obj.getElementsByTagName("Annotation")
    for i in range(len(Annotation)):
        if Annotation[i].getAttribute("Id") == regionId: # ="1":
            Regions=Annotation[i].getElementsByTagName("Regions")
            Region=Regions[0].getElementsByTagName("Region")
    
            for j in range(len(Region)):
                Vertex=Region[j].getElementsByTagName("Vertex")
                hold=[]
                for k in range(len(Vertex)-1):
                    endpoints=np.zeros(shape=(1,2))
                    endpoints[0,0] = Vertex[k].getAttribute("X")
                    endpoints[0,1] = Vertex[k].getAttribute("Y")
                    hold.extend(endpoints)
                closedline = np.round(np.array([hold])*rate)
                if closedline.size >=5:
                    cv2.fillPoly(canvas,closedline.astype(np.int64), 255)
    return canvas

def GetClosedline(Region,rate=1):
    Vertex=Region.getElementsByTagName("Vertex")
    hold=[]
    for k in range(len(Vertex)-1):
        endpoints=np.zeros(shape=(1,2))
        endpoints[0,0] = Vertex[k].getAttribute("X")
        endpoints[0,1] = Vertex[k].getAttribute("Y")
        hold.extend(endpoints)
    closedline = np.round(np.array([hold])*rate)
    return closedline
            
#-----------------------------------------------------------------------------#
def load_xlsx(path,sheetid):
    data = xlrd.open_workbook(path)
    table = data.sheets()[sheetid] # 0,1,2...
    nrows = table.nrows
    info = []
    for i in range(nrows):
        info.append(table.row_values(i)[:])
    return info

def convertLabel(labelpath,hospitalname,key):
    if hospitalname == '6-ROI':
        info = load_xlsx(labelpath,1)[1:]
        labelindex = 23
    elif hospitalname == 'CC-ROI':
        info = load_xlsx(labelpath,0)[1:]
        labelindex = 25
    for i in range(0,len(info)):
        if re.search(key,info[i][0]):
            return int(float(info[i][labelindex]))
        else:
#            print(i,' error, ',idlist[idx])
            continue
#-----------------------------------------------------------------------------#
from keras.applications.densenet import DenseNet201
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D,Flatten
from keras import optimizers
from keras.models import Model
from keras.callbacks import EarlyStopping,TensorBoard,ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator

def CNN_model_VGG19_pretrained(insize = [224,224,3],weightspath=None):
    base_model = VGG19(include_top=False,
              weights=None,
              input_tensor=None,
              input_shape=insize,
              pooling=None,
              classes=9)
     
    x = base_model.output
    #x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    predictions = Dense(9, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    if weightspath is not None:
        model.load_weights(weightspath) 
    return model         
def CNN_model_VGG19(insize = [224,224,3],weightspath=None):
    model = VGG19(include_top=True,
              weights=weightspath,
              input_tensor=None,
              input_shape=insize,
              pooling=None,
              classes=9)
    return model 
        
#-----------------------------------------------------------------------------#


#main_path =  'D:\\Data\\RectumCancer'
main_path =  './data/HEsvs'
#labelpath = '../data/Info/Patients Information(MRI-p).xlsx'
#hospitallist = ['6-ROI','CC-ROI']
hospitallist = ['6-ROI']
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'

for hospitalname in hospitallist:
    idlist = sorted(os.listdir(os.path.join(main_path,hospitalname)))
#    for idx in range(0,len(idlist)):
    for idx in range(0,1):
        idname = idlist[idx].split('_')[0]
        print(hospitalname,idx,idlist[idx],idname)
        imgPath = os.path.join(os.path.join(main_path,hospitalname,idlist[idx],idname+'.svs'))
        xmlPath = os.path.join(os.path.join(main_path,hospitalname,idlist[idx],idname+'.xml'))
#        imgPath = os.path.join(os.path.join(main_path,hospitalname,idlist[idx],'p',idname+'.svs'))
#        xmlPath = os.path.join(os.path.join(main_path,hospitalname,idlist[idx],'p',idname+'.xml'))
#        label = convertLabel(labelpath ,hospitalname,idlist[idx])# label
        # load .svs
        slide = openslide.OpenSlide(imgPath)
        
        downsamples=slide.level_downsamples #每一个级别K的对应的下采样因子，下采样因子应该对应一个倍率
        print('Origin size ',slide.level_dimensions,' ,downsamples',downsamples)
        [w, h] = slide.level_dimensions[0] #最高倍下的宽高
#        resizerate = downsamples[0]/downsamples[1]
#        resizerate = 1
#        size1 = int(w*resizerate)# 计算级别k下的总宽
#        size2 = int(h*resizerate) # 计算k下的总高
        
        win_size = 224
        batch_size = 64
        batchnum = 0
        tmp_start = []
        x40_start = []
        cnn_pred_clss = []
        cnn_pred_score = []       
        step_m = win_size - win_size//6
        step_n = win_size - win_size//6
        # Build CNN model
#        model = CNN_model_VGG19_pretrained(insize = [224,224,3],weightspath = './model/model.h5') #pretrained
        model = CNN_model_VGG19(insize = [224,224,3],weightspath = './model/model.h5')
        for m in range(0,h,step_m):
            for n in range(0,w,step_n):
                start = [n, m] # width height
                region = np.array(slide.read_region((start[0],start[1]), 0, (win_size, win_size)))[:,:,:3]
#                if np.std(region)>10 and batchnum<batch_size:
                if batchnum<batch_size:
                    tmp_start.append(start) # n*2
                    if batchnum == 0 :
                        inputs = np.expand_dims(region,axis=0)
                    else:
                        inputs = np.concatenate((inputs,np.expand_dims(region,axis=0)),axis=0)
                    batchnum = batchnum+1
                    
                if batchnum == batch_size:
                    print('total patch:',(h//step_m)*(w//step_n),' ,Processing: ',
                          len(x40_start),' ,Complete rate: ',(len(x40_start))/((h//step_m)*(w//step_n)))
                    batchnum=0
                    tmp_results = model.predict(inputs)
                    x40_start.extend(tmp_start)
                    cnn_pred_clss.extend(list(np.argmax(tmp_results,axis=1)))
                    cnn_pred_score.extend(list(np.max(tmp_results,axis=1)))
                    tmp_start = []
                    
#                    sys.exit()   
        
        resultsDir = './results'
        mkdir(os.path.join(resultsDir,idname))  
        np.savetxt(os.path.join(resultsDir,idname,idname+'_'+'m'+str(step_m)+'_n'+str(step_n)+'_x40_start.csv'),np.array(x40_start), delimiter=',')
        np.savetxt(os.path.join(resultsDir,idname,idname+'_'+'m'+str(step_m)+'_n'+str(step_n)+'_cnn_pred_clss.csv'),np.array(cnn_pred_clss), delimiter=',')
        np.savetxt(os.path.join(resultsDir,idname,idname+'_'+'m'+str(step_m)+'_n'+str(step_n)+'_cnn_pred_score.csv'),np.array(cnn_pred_score), delimiter=',')
        # convertTomask
        colors = np.array([[141,141,141],[53,53,53],[189,41,153], [30,149,191],
                           [250,216,206],[68,172,35],[204,102,0],[247,188,10],[230,73,22]])
        pointPos = np.array(x40_start)//step_m # canvas coordinate
#        canvas = np.zeros([np.max(pointPos[:,1]),np.max(pointPos[:,0])],dtype=np.uint8)
        canvas = np.zeros([h//step_m+1,w//step_n+1,3],dtype=np.uint8)
        clssmap = np.zeros([h//step_m+1,w//step_n+1,3])
        scoremap = np.zeros([h//step_m+1,w//step_n+1,3])
        for p_idx in range(pointPos.shape[0]):
            canvas[pointPos[p_idx][1],pointPos[p_idx][0],:] = colors[cnn_pred_clss[p_idx]]
            clssmap[pointPos[p_idx][1],pointPos[p_idx][0],:] = cnn_pred_clss[p_idx]
            scoremap[pointPos[p_idx][1],pointPos[p_idx][0],:] = cnn_pred_score[p_idx]
        imageio.imwrite(os.path.join(resultsDir,idname,idname+'_'+'m'+str(step_m)+'_n'+str(step_n)+'_canvas.tif'),canvas)
        imageio.imwrite(os.path.join(resultsDir,idname,idname+'_'+'m'+str(step_m)+'_n'+str(step_n)+'_clssmap.tif'),clssmap)
        imageio.imwrite(os.path.join(resultsDir,idname,idname+'_'+'m'+str(step_m)+'_n'+str(step_n)+'_scoremap.tif'),scoremap)
            
    

