import streamlit as st 
import cv2
import numpy as np
import face_recognition
import dlib
import cmake
import os
from datetime import datetime

st.title("Shashwath")
st.write("Shashwath is a pltform that helps Agniveers to connect to real world.")

path = r"C:\Users\nisha\OneDrive\Desktop\LOC\Attendance\Attendance\Database"
images = []
classnames = []
mylist = os.listdir(path)
print(mylist)


for i in mylist:
    #
    curImg = cv2.imread(f'{path}/{i}')
    
    images.append(curImg)
    classnames.append(os.path.splitext(i)[0])



def findencoding(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markattendance(name):
    with open("Attendance.csv","r+") as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(",")
            namelist.append(entry[0])

        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dtstring}')


    

encodeknown = findencoding(images)



cap =st.camera_input("Take a picture to know your scores")
if cap is not None:
    
#while True:
    '''ret, frame = cap.read()
    frameS = cv2.resize(frame,(0,0),None,0.25,0.25)
    frameS = cv2.cvtColor(frameS,cv2.COLOR_BGR2RGB)'''

    location = face_recognition.face_locations(cap)
    encode = face_recognition.face_encodings(cap,location)

    for encodeface, facelocation in zip(encode,location):
        matches = face_recognition.compare_faces(encodeknown, encodeface)
        face_dis = face_recognition.face_distance(encodeknown, encodeface)
        matchindex = np.argmin(face_dis)


        if matches[matchindex]:
            name = classnames[matchindex].upper()
            markattendance(name)
    key=cv2.waitKey(100)

cap.release()

#st.metric("Height",5.5)
#st.button(label="Job Suggestions")
#st.success("JOBS")