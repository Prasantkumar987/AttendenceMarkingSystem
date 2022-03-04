import streamlit as st
import cv2 
from mtcnn.mtcnn import MTCNN as mt
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
from webcam import webcam
from datetime import datetime
import os
import csv
import pandas as pd


from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

def attendence_punching(result):
    now = datetime.now()
    lastAttendanceTime = now.replace(hour=20, minute=0, second=0, microsecond=0)
    current_time = now.strftime("%H:%M")
    current_date = datetime.today().strftime('%Y-%m-%d')
    # print(current_date)
    lastTime= lastAttendanceTime.strftime("%H:%M")
    if (current_time < lastTime):
        
        if not os.path.isfile(f"./attendence/{current_date}.csv"):
            with open(f"./attendence/{current_date}.csv", 'w') as f: 
                w = csv.writer(f) 
                w.writerow(['E_Id','Attendence Punching time', 'Status']) 
                w.writerow([result[0],current_time,'Present']) 
                st.success(f"{result[0]}, your attendence is successfully marked at {current_time}.")
                
        else:
            with open(f"./attendence/{current_date}.csv", 'a') as f: 
                w = csv.writer(f) 
                df= pd.read_csv(f"./attendence/{current_date}.csv")
                df['E_Id']= df['E_Id'].astype('object')
                print(df.dtypes)
                print(type(result[0]))
                print(type(df['E_Id'][0]))
                temp_df= df[df['E_Id']== int(result[0])]
                print('Dataframe', temp_df)
                print("Present or Not",(temp_df[temp_df['Status']== 'Present']))
                temp_df= temp_df[temp_df['Status']== 'Present']
                if temp_df.empty:
                    df.loc[df.E_Id== int(result[0]), "Attendence_Punching_time"] = current_time
                    df.loc[df.E_Id== int(result[0]), 'Status'] = 'Present' 
                    st.success(f"{result[0]}, your attendence is successfully marked at {current_time}.")
                    df.to_csv(f'./attendence/{current_date}.csv')
                else:
                    # print(result[0] in df.E_Id )
                    st.success(f"{result[0]}, your attendence is already marked.")
                
    else:
        st.warning(f"{result[0]}, you are late to mark your attendence. Last time for marking attendence is {lastTime} AM")


def csv_maintain():
    current_date = datetime.today().strftime('%Y-%m-%d')
    emp= ['0000001', '0000255', '0000262', '0000264', '0000265', '0000268', '0000274', '0000278', '0000281', '0000283', '0000284', '0000286', '0000292', '0000293', '0000297', '0000298', '0000299', '0000301', '0000303', '0000304']    
    if not os.path.isfile(f"./attendence/{current_date}.csv"):
            with open(f"./attendence/{current_date}.csv", 'w') as f: 
                w = csv.writer(f) 
                w.writerow(['E_Id','Attendence Punching time', 'Status']) 
                w.writerow([str(emp[0]),"","Absent"]) 
    else:   
        for i in range(1, len(emp)):
            with open(f"./attendence/{current_date}.csv", 'a') as f: 
                w = csv.writer(f)
                w.writerow([str(emp[i]),"","Absent"])  



def load_image(image_file):
	img = Image.open(image_file)
	return img


def bbox(img, result):
    detector= mt()
    faces= detector.detect_faces(img)
    
    for i in range(0, len(faces)):
        [x,y,w,h] = faces[i]['box']
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),5)
        cv2.putText(img, '{} : {:.2f}'.format(result[0],result[1]), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)  
    #img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    


def face_match(img): # img_path= location of photo, data_path= location of data.pt 
    # getting embedding matrix of the given img
    
    face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
    emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
    
    saved_data = torch.load('./data.pt') # loading data.pt file
    embedding_list = saved_data[0] # getting embedding data
    name_list = saved_data[1] # getting list of names
    dist_list = [] # list of matched distances, minimum distance is used to identify the person
    
    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)
        
    idx_min = dist_list.index(min(dist_list))
    return (name_list[idx_min], min(dist_list))

    # print('Face matched with: ',result[0], 'With distance: ',result[1])
    # bbox(image, result)
def attendence(img_file):
    # if img_file is not None:
    img= load_image(img_file)
    result= face_match(img)
    print(result)
    cv_img = np.array(img)
    bounded_face= bbox(cv_img, result)
    st.image(bounded_face,channels="RGB",width= 350)
    attendence_punching(result)
    print('Face matched with: ',result[0], 'With distance: ',result[1])


st.title('Attendence Checking System')
items= ['Capture Image', 'Upload Image']
ch= st.sidebar.selectbox(" Image",items)

if ch == 'Capture Image':
    # class VideoTransformer(VideoTransformerBase):
    #     def __init__(self):
    #         self.i = 0

    #     def transform(self, frame):
    #         # img = frame.to_ndarray(format="bgr24")
    #         img= load_image(frame)
    #         print("Image getting Loaded")
    #         result= face_match(img)
    #         print(result)
    #         cv_img = np.array(frame)
    #         bbox(cv_img, result)

    #         return img
    # webrtc_streamer(key= 'attendence',video_transformer_factory=VideoTransformer)
    
        # while 0:
    #     ret, img = cap.read()
    #     img_display.image(img, channels= 'BGR')
    #     b= st.button('Click Image')
    #     if b:
    #         result= face_match(img)
    #         bbox(load_image(img), result)
    #         print('Face matched with: ',result[0], 'With distance: ',result[1])
    #         cap.release()

    captured_image = webcam()
    # csv_maintain()
    if captured_image is None:
        st.write("Waiting for capture...")
    else:
        st.write("Got an image from the webcam:")
        # print(type(captured_image))
        # st.image(captured_image)
        background = Image.new("RGB", captured_image.size, (255, 255, 255))
        background.paste(captured_image, mask = captured_image.split()[3])

        result= face_match(background)
        # print(np.shape(captured_image))
        # print(captured_image)
        print(result)
        
        cv_img = np.array(background)
        bounded_face= bbox(cv_img, result)
        st.image(cv_img,channels="RGB",width= 640)
        attendence_punching(result)

        # attendence(captured_image)

else:
    st.subheader('Upload Image')
    img_file= st.file_uploader('img',type=['png','jpg','jpeg'])
    print(type(img_file))
    attendence(img_file)





