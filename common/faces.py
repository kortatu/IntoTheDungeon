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

        self.addNewImages()


    def getImages(self):
        if random.randint(1, 100) > 95: # 5% change we recheck the folder
            self.addNewImages()
        return { 'images': self.encodings, 'names': self.names }

    def addNewImages(self):

        onlyfiles = [f for f in listdir(self.dir) if (isfile(join(self.dir, f)) and ".jpg" in f )]
        print(onlyfiles)
        print(self.dir)
        for pic in onlyfiles:
            name = pic.replace(".jpg", "")
            if (name not in self.names):
                print "Processing " + name
                image = face_recognition.load_image_file(self.dir + "/" + pic)
                image_encoding = face_recognition.face_encodings(image)[0]
                self.encodings.append( image_encoding )
                self.names.append( name )