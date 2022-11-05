import cv2
import numpy as np
import face_recognition
import dlib
import cmake
import os
from datetime import datetime

path = r"C:\Users\Rishabh\Desktop\WORK\hackathons\LOC\Attendance\DATABASE"
images = []
classnames = []
mylist = os.listdir(path)
print(mylist)


for i in mylist:
    curImg = cv2.imread(f'{path}/{i}')
    images.append(curImg)
    classnames.append(os.path.splitext(i)[0])
print(classnames)


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
print("Encoding Complete")


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frameS = cv2.resize(frame,(0,0),None,0.25,0.25)
    frameS = cv2.cvtColor(frameS,cv2.COLOR_BGR2RGB)
    
    location = face_recognition.face_locations(frameS)
    encode = face_recognition.face_encodings(frameS,location)

    for encodeface, facelocation in zip(encode,location):
        matches = face_recognition.compare_faces(encodeknown, encodeface)
        face_dis = face_recognition.face_distance(encodeknown, encodeface)
        matchindex = np.argmin(face_dis)


        if matches[matchindex]:
            name = classnames[matchindex].upper()
            #print(name)
            y1,x2,y2,x1 = facelocation
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
            markattendance(name)
    cv2.imshow("Image",frame)
    cv2.waitKey(1)
