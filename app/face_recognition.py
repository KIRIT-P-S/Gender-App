import numpy as np
import sklearn
import pickle
import cv2
import os
haar=cv2.CascadeClassifier(r'C:\Users\DELL\Facerecognitionproject\FaceRecognition_Flask-main\4_Flask_App\model\haarcascade_frontalface_default.xml')
model_svm=pickle.load(open(r'C:\Users\DELL\Facerecognitionproject\FaceRecognition_Flask-main\4_Flask_App\model\model_svm.pickle',mode='rb'))
pca_models=pickle.load(open(r'C:\Users\DELL\Facerecognitionproject\FaceRecognition_Flask-main\4_Flask_App\model\pca_dict.pickle',mode='rb'))
model_pca=pca_models['pca']
mean_face_arr=pca_models['mean_face']

def faceRecognitionPipeline(filename, path=True):
    if path:
        # Check if the file exists
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' does not exist.")
            return None, None
        img = cv2.imread(filename)  # Read image from file
        if img is None:
            print("Error: cv2.imread failed to read the file.")
            return None, None
    else:
        img = filename  # Use directly if it's a frame

    # step-02: convert into gray scale
    gray =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    # step-03: crop the face (using haar cascase classifier)
    faces = haar.detectMultiScale(gray,1.5,3)
    predictions = []
    for x,y,w,h in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi = gray[y:y+h,x:x+w]
        
        # step-04: normalization (0-1)
        roi = roi / 255.0
        # step-05: resize images (100,100)
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
            
        # step-06: Flattening (1x10000)
        roi_reshape = roi_resize.reshape(1,10000)
        # step-07: subtract with mean
        roi_mean = roi_reshape - mean_face_arr # subtract face with mean face
        # step-08: get eigen image (apply roi_mean to pca)
        eigen_image = model_pca.transform(roi_mean)
        # step-09 Eigen Image for Visualization
        eig_img = model_pca.inverse_transform(eigen_image)
        # step-10: pass to ml model (svm) and get predictions
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
        
        # step-11: generate report
        text = "%s : %d"%(results[0],prob_score_max*100)
        # defining color based on results
        if results[0] == 'male':
            color = (255,255,0)
        else:
            color = (255,0,255)
            
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color,-1)
        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),5)
        output = {
            'roi':roi,
            'eig_img': eig_img,
            'prediction_name':results[0],
            'score':prob_score_max
        }
        
        predictions.append(output)
    return img, predictions