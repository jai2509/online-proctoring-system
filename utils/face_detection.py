import cv2
import dlib

detector = dlib.get_frontal_face_detector()

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces
