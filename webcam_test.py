import face_recognition
import cv2

from os import listdir
from os.path import isfile, join

dir = "dragontrainer/trainImages/pics"

video_capture = cv2.VideoCapture(0)

onlyfiles = [f for f in listdir(dir) if (isfile(join(dir, f)) and ".jpg" in f )]

encodings = [];
names = [];

for pic in onlyfiles:
    name = pic.replace(".jpg", "")
    print "Processing " + name
    image = face_recognition.load_image_file(dir + "/" + pic)
    image_encoding = face_recognition.face_encodings(image)[0]
    encodings.append( image_encoding )
    names.append( name )

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:

    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(encodings, face_encoding)
            name = "Unknown"

            for index, val in enumerate(match):
                if(val):
                    name = names[index]

            face_names.append(name)


    process_this_frame = not process_this_frame


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


    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()