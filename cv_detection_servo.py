import cv2
import numpy as np
import time

classes = []
clsfile = 'obj.names'
with open(clsfile) as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet('yolov3-tiny-custom_last.weights','yolov3-tiny-custom.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

dispW=640
dispH=480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)

font = cv2.FONT_HERSHEY_PLAIN
new_time = 0
prev_time = 0
frame_id = 0
conftresh = 0.5
NMStresh = 0.5

colors = [(0,255,0),(0,0,255)]

while True:
    _,frame = cap.read()
    frame_id = 1
    
    
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>input and detection
    height,width,channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame,1/255,(320,320),(0,0,0),True,crop = False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conftresh :
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x=int(center_x - w/2)
                y=int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,conftresh,NMStresh)
    
    #>>>>>>>>>>>>>>>>>>>>>>>>>>bounding box
    obj_id = 4
    for i in range(len(boxes)):
        if i in indexes :
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            obj_id = class_ids[i]
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame, label+str(round(confidence,2)),(x,y+30),font,1,(266,266,266),2)

    new_time = time.time()
    fps=new_time - prev_time
    prev_time = time.time()

    cv2.putText(frame, "FPS"+str(round(fps,2)),(10,50),font,1,(0,0,0),1)

    cv2.imshow("image",frame)
    
    #>>>>>>>>>>>>>>>>>> creating text file for servo control
    with open ('detection_id.txt','w') as file:
        file.write(str(obj_id))
        file.close()
    
    #>>>>>>>>>>>>>>>>>> exit detection
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
