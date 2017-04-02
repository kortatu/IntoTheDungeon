FROM library/python:2-onbuild
VOLUME /usr/src/app/dragon/tmp
VOLUME /usr/src/app/saves
VOLUME /usr/src/app/dragontrainer/trainImages
VOLUME /usr/src/app/dragontrainer/latest
EXPOSE 5001
CMD [ "python", "/usr/src/app/dragon/main.py"]