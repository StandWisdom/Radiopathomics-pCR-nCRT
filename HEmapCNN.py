import os,re,shutil,h5py
import numpy as np 
import tifffile
import matplotlib.pyplot as plt

#def generateTrain(path):
#    idlist = next(os.walk(path))[1]
#    for idx in range(len(idlist)):
#        filelist = next(os.walk(os.path.join(path,idlist[idx],'predmap')))[2]
#        for filename in filelist:
#            tif = tifffile.imread(os.path.join(path,idlist[idx],'predmap',filename))
#            print(tif.shape) 
#            if tif.shape().any
#            
#    return

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\") 
    isExists=os.path.exists(path) 
    if not isExists:
        os.makedirs(path)  
        print ('create folder:' +path+' complete!')
        return True
    else:
        return False 
    
def generateTrain(path):
    #savepath = './data/HE9map'
    #for i in range(4):
    #    mkdir(os.path.join(savepath,str(i)))
    idlist = next(os.walk(path))[1]
    clss = []
    for idx in range(0,len(idlist)):
    #for idx in range(0,1):
        filelist = next(os.walk(os.path.join(path,idlist[idx],'predmap')))[2]
        for filename in filelist:
            print(idx,idlist[idx],filename)
            tif = tifffile.imread(os.path.join(path,idlist[idx],'predmap',filename))
    #        print(tif.shape) 
    #        dim = tif.shape
            size = 224
            if tif.shape[0]<size and tif.shape[1]<size:
                sample = np.zeros([size,size,9])
                sample[0:tif.shape[0],0:tif.shape[1],:] = tif
            elif tif.shape[0]<size:
                sample = np.zeros([size,size,9])
                sample[0:tif.shape[0],0:sample.shape[1],:] = tif[:,0:sample.shape[1],:]
            elif tif.shape[1]<size:
                sample = np.zeros([size,size,9])
                sample[0:sample.shape[0],0:tif.shape[1],:] = tif[0:sample.shape[0],:,:]
            else:
                sample = np.zeros([size,size,9])
                sample = tif[0:size,0:size,:]
            clss.append(int(idlist[idx].split('_')[1]))
            if 'data' not in locals().keys():        
                data = np.expand_dims(sample,axis=0)
            else:
                data = np.concatenate((data,np.expand_dims(sample,axis=0)),axis=0)
    
    h5path = './data/HE_h5'
    h5 = h5py.File(os.path.join(h5path,'my.h5'),'w')
    h5['data'] = data
    h5['label'] = clss
    h5.close()
#------------------------------------------------------------------------------#
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet121
from keras import optimizers
from keras.callbacks import EarlyStopping,TensorBoard,ModelCheckpoint 
from sklearn.preprocessing import LabelBinarizer  
#path = './newresults/new_results_6'
#generateTrain(path)
#    
h5path = './data/HE_h5'
h5 = h5py.File(os.path.join(h5path,'my.h5'),'r')
data = h5['data'][:]
label = h5['label'][:]
h5.close()
print(data.shape)
# prepare
label_0 = np.where(label==0)
label_1 = np.where(label!=0)
label[label_0]=1
label[label_1]=0
encoder = LabelBinarizer()
label = encoder.fit_transform(label)
 
# build model
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#model = VGG19(include_top=True,
#          weights=None,
#          input_tensor=None,
#          input_shape=[299,299,9],
#          pooling=None,
#          classes=4)
model = DenseNet121(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=[224,224,9],
                pooling=None,
                classes=1)
model.compile(optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),loss='binary_crossentropy',\
              metrics=['accuracy'])#categorical_crossentropy
early_stopping = EarlyStopping(monitor='val_acc', patience=10)
mytensorboard = TensorBoard(log_dir='./model')
checkpoint = ModelCheckpoint(filepath='./model/model_checkpoint.h5',monitor='val_acc',mode='auto' ,save_best_only='True')
#callback_lists=[mytensorboard,checkpoint,early_stopping]        
callback_lists=[early_stopping] 
# Train                   
hist = model.fit(data,label,epochs=100,batch_size=16,callbacks=callback_lists,validation_split=0.1,shuffle=True)
# save model
#if hist:
#    print('Saving model to .h5...')
#    model.save('./model/HEmap_model_final.h5')
#    with open('./model/log.txt','w') as f:
#        f.write(str(hist.history))
