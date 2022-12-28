#Import the Open-CV extra functionalities
import cv2
import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

##two motors attached...
#dog forward
DOG1= 23
#dog backward
DOG2=1
CAT2=3
CAT1=24

GPIO.setup(DOG1, GPIO.OUT)
GPIO.setup(CAT1, GPIO.OUT)
GPIO.setup(DOG2, GPIO.OUT)
GPIO.setup(CAT2, GPIO.OUT)

GPIO.output(DOG1, GPIO.LOW)
GPIO.output(CAT1, GPIO.LOW)
GPIO.output(DOG2, GPIO.LOW)
GPIO.output(CAT2, GPIO.LOW)

#This is to pull the information about what each object is called
classNames = []
classFile = "file.txt"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

#This is to pull the information about what each object should look like
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

doggo = False
kitty = False

#This is some set up values to get good results
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def dog():
    doggo=True
    time.sleep(10)
    GPIO.output(DOG1,GPIO.HIGH)
    GPIO.output(DOG2, GPIO.HIGH)
    time.sleep(30)
    GPIO.output(DOG1,GPIO.LOW)
    GPIO.output(DOG2, GPIO.LOW)
    time.sleep(3)

def cat():
    kitty=True
    time.sleep(10)
    GPIO.output(DOG2, GPIO.LOW)
    GPIO.output(CAT1, GPIO.HIGH)
    GPIO.output(CAT2, GPIO.HIGH)
    time.sleep(1)
    GPIO.output(CAT1,GPIO.LOW)
    GPIO.output(CAT2, GPIO.LOW)
    time.sleep(3)

#This is to set up what the drawn box size/colour is and the font/size/colour of the name tag and confidence label   
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo

#Below determines the size of the live feed window that will be displayed on the Raspberry Pi OS
if _name_ == "_main_":

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    
    #Below is the never ending loop that determines what will happen when an object is identified.    
    while True:
        success, img = cap.read()
        #Below provides a huge amount of controll. the 0.65 number is the threshold number, the 0.2 number is the nms number)
        result, objectInfo = getObjects(img,0.45,0.2)
        
        if(objectInfo!=[]):
            
            if(objectInfo[0][1]=="animalcat 17" and kitty==False):
                print("it a kitty cat")
                cat()
                objectInfo == []
                time.sleep(2)
                objectInfo != []
                
            elif(objectInfo[0][1]=="animaldog 18" and doggo==False):
                print("it a doggo")
                dog()
                objectInfo == []
                time.sleep(2)
                objectInfo != []
                
                            

        cv2.imshow("Output",img)
        cv2.waitKey(1)