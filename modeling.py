import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,roc_curve,auc

#-----------------------------------------------------------------------------#         
def ResultsOfSVM(X_train, y_train, X_test, y_test,c_Value=-2,g_value = 1,keywords=None):
    clf = SVC(C=2 ** c_Value, kernel='rbf', gamma=2 ** g_value, class_weight='balanced',
              probability=True)  # , class_weight={1:classWeight, 0:1}, class_weight='balanced'
    SVM_model = clf.fit(X_train, y_train)
    train_score = SVM_model.predict_proba(X_train)
    test_score = SVM_model.predict_proba(X_test)
    train_pred = SVM_model.predict(X_train)
    test_pred = SVM_model.predict(X_test)
    # 计算AUC找到合适的C值
    fpr0, tpr0, _ = roc_curve(y_train, train_score[:, 1])
    aucTrain = auc(fpr0, tpr0)
    fpr1, tpr1, _ = roc_curve(y_test, test_score[:, 1])
    aucTest = auc(fpr1, tpr1)
#    print(aucTrain,aucTest)
    
    plt.figure(figsize=(5,5))
    plt.title(keywords+'\n Train AUC:'+str(round(aucTrain,3))+' Test AUC:'+str(round(aucTest,3)))
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.plot(fpr0, tpr0, color='red',label='train')
    plt.plot(fpr1, tpr1, color='blue',label='test')  #蓝
    plt.legend(loc = 'lower right')
    plt.show()
    return train_score,test_score,train_pred,test_pred

if __name__ == '__main__':
    print('svm')