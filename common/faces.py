import face_recognition
import cv2
import random

from os import listdir
from os.path import isfile, join

#
# Class to handle first load of images, and checks if new images were added
#
class Faces(object):
    dir = ""

    def __init__(self, dir):
        self.encodings = []
        self.names = []
        self.dir = dir
        self.imageObjects = []
        self.addNewImages()


    def getImages(self):
        if random.randint(1, 100) > 95: # 5% change we recheck the folder
            self.addNewImages()
        return { 'images': self.encodings, 'names': self.names, 'objects': self.imageObjects }

    def addNewImages(self):

        onlyfiles = [f for f in listdir(self.dir) if (isfile(join(self.dir, f)) and ".jpg" in f )]
        for pic in onlyfiles:
            name = pic.replace(".jpg", "")
            if (name not in self.names):
                print ("Processing " + name)
                try:
                    image = face_recognition.load_image_file(self.dir + "/" + pic)
                    faces = face_recognition.face_encodings(image)
                    landmarks = face_recognition.face_landmarks(image)
                    if faces:
                        for index, face in enumerate(faces):
                            self.encodings.append( face )
                            self.names.append( name + "_" + str(index)  if index > 0 else name )
                            imageObject = {
                                'encoding': face,
                                'name': name + "_" + str(index)  if index > 0 else name,
                                'landmarks': landmarks[index],
                                'file': pic,
                                'dir' : self.dir,
                                'path': self.dir + "/" + pic,
                                'image': image,
                            }
                            self.imageObjects.append(imageObject)

                except IOError:
                    pass
