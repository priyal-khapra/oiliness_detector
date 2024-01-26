import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array

def face_extractor(img):
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        h,w,_ = img.shape 
        face_detection_results = face_detection.process(img[:,:,::-1])
        if face_detection_results.detections:
            for face_no, face in enumerate(face_detection_results.detections):
                face_data = face.location_data
                faces =[[int(face_data.relative_bounding_box.xmin*w),int(face_data.relative_bounding_box.ymin*h),
                         int(face_data.relative_bounding_box.width*w),int(face_data.relative_bounding_box.height*h)]]
                for x, y, w, h in faces:
                    cropped_img = img[y-int(h/3):y + 13*int(h/12), x:x + w]
        if cropped_img is None:
            print("Could not read input image")
    except Exception as e:
        cropped_img = False
    return cropped_img

def landmark_oiliness(img):
    oiliness = [[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,397, 365, 379, 378, 400,
                 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
                 [57, 185, 40, 39, 37, 0, 267, 269, 270, 409, 287, 375, 321, 405, 314, 17, 84, 181, 91, 146, 146, 146, 146, 146,
                 146, 146, 146, 146, 146, 146, 146, 146, 146, 146, 146, 146]]
    list_landmarks_forehead = []
    face_cropped_img = img.copy()
    h,w,_ = face_cropped_img.shape
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                        min_detection_confidence=0.5)
    face_mesh_results = face_mesh_images.process(face_cropped_img[:,:,::-1])
    if face_mesh_results.multi_face_landmarks:
        for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            for ind in oiliness:
                list_landmark=[]
                for pnt in ind:
                    list_landmark.append([int(face_landmarks.landmark[pnt].x*w),
                                            int(face_landmarks.landmark[pnt].y*h)])
                list_landmarks_forehead.append(list_landmark) 
    return list_landmarks_forehead

def face_masking(face_cropped_img,list_landmarks):
    try:
        face = face_cropped_img.copy()
        mask = np.zeros((face_cropped_img.shape[0],face_cropped_img.shape[1]))
        cv2.drawContours(mask, np.array(list_landmarks), -1, 255, -1)
        mask = mask.astype(np.bool_)
        masked_face = np.zeros_like(face)
        masked_face[mask] = face[mask]
    except Exception as e:
        masked_face = False
    return masked_face  

def heatmapping(img):
    colormap = plt.get_cmap('inferno')
    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
    return heatmap 

def hsv_masking(img):
    hsv = cv2.cvtColor(heatmapping(img), cv2.COLOR_BGR2HSV)
    Lower_hsv = np.array([140,0,0])
    Upper_hsv = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
    return mask 

def oiliness_viz(img):
    try:
        face = face_extractor(img)
        landmark = landmark_oiliness(face)
        mask= face_masking(face,landmark)
        hsv= hsv_masking(mask)
        overlay = face.copy()
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,(0,255,255),-1)
        blur = cv2.GaussianBlur(overlay, (13,13), 0)
        alpha = 0.2 
        new_img = cv2.addWeighted(blur, alpha, face, 1 - alpha, 0)
        mask_area= np.sum(hsv != 0)
        total_area= np.sum(mask != 0)/10
        perc = (mask_area/total_area)*10
        result= round(perc,2)
        #result= result *10
        if result>10:
            result=10
        if result < 4:
            oil_class= "Good"
        elif result >6:
            oil_class= "Bad"
        else:
            oil_class= "Average"
    except Exception as e:
        print(e)
        new_img, result, oil_class = False, False, False
    return new_img, result, oil_class


img = cv2.imread("C:/Users/priya/OneDrive/Desktop/mlh/image2.jpg")
oiliness_img_ann, oil_score, oil_class = oiliness_viz(img)
print(oil_class)
print(oil_score)
cv2.imshow("image", oiliness_img_ann)
cv2.waitKey(0)
cv2.destroyAllWindows()