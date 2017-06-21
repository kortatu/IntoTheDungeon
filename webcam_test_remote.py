import face_recognition
import cv2
import urllib
import common.faces as faces
import numpy as np
import subprocess

dir = "dragontrainer/trainImages/pics"
# print("Opening subprocess")
# subprocess.Popen(["python", "imshow.py"])
# print("Opened subprocess")

# video_capture = cv2.VideoCapture("http://192.168.0.68:8090/webcam.mpg", cv2.CAP_FFMPEG)
# video_capture = cv2.VideoCapture("http://192.168.0.68:8090/webcam.mpg")
video_capture=urllib.urlopen('http://192.168.0.68:8090/webcam.mpg')
print("Opened video_caputre")
facesLib = faces.Faces(dir)
print("Init faces lib")

encodings = facesLib.getImages()['images']
names = facesLib.getImages()['names']

face_locations = []
face_encodings = []
face_names = []
process_this_frame = 0


# std::string address = "rtsp://<username:password>@<ip_address>:<port>";
# cv::VideoCapture cap;
#
# if(!cap.open(address))
# {
#     std::cout << "Error opening video stream: " << address << std::endl;
# return -1;
# }
bytes=''

while True:
    # ret, frame = video_capture.read()
    bytes+=video_capture.read(1024)

    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    print("Into the loop")

    if a!=-1 and b!=-1:
        print("Found the bytes")

        jpg = bytes[a:b+2]
        bytes= bytes[b+2:]
        if process_this_frame % 5 == 0:
            frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)

            #Refresh in case we have new faces
            encodings = facesLib.getImages()['images']
            names = facesLib.getImages()['names']

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            print("resizedFrame")
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(encodings, face_encoding)
                name = "Unknown"
                maxScore = 99

                for index, val in enumerate(distances):
                    if(val <= 0.6 and val < maxScore):
                        maxScore = val
                        name = names[index]

                face_names.append(name)


            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/2 size
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                if(name == "Unknown"):
                    cv2.line(frame,(right, bottom),(left,top),(0,0, 255),3)
                    cv2.line(frame,(right, top),(left,bottom),(0,0, 255),3)


                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                print(name)

            cv2.imshow('i', frame)
            cv2.waitKey(1)
        process_this_frame = process_this_frame + 1



video_capture.release()
cv2.destroyAllWindows()
