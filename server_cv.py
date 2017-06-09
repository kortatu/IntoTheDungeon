import socket
import cv2
import pickle
import struct
import face_recognition

from os import listdir
from os.path import isfile, join


dir = "dragontrainer/trainImages/pics"
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

HOST=''
PORT=8089

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print 'Socket created'

s.bind((HOST,PORT))
print 'Socket bind complete'
s.listen(10)
print 'Socket now listening'

conn,addr=s.accept()

face_locations = []
face_encodings = []
face_names = []

### new
data = ""
payload_size = struct.calcsize("I")
while True:
    while len(data) < payload_size:
        data += conn.recv(4096)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("I", packed_msg_size)[0]
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    ###

    frame=pickle.loads(frame_data)

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(encodings, face_encoding)
        name = "Unknown"

        for index, val in enumerate(match):
            if(val):
                name = names[index]

        face_names.append(name)

    print(face_locations)
    print(face_names)
