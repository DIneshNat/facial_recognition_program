import face_recognition
import cv2
import numpy as np


# define a video capture object
vid_cap = cv2.VideoCapture(0)

#recognizing sample image setup
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

trump_image = face_recognition.load_image_file("trump.jpg")
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

#Array of known faces and namnes
known_face_encodings = [
    obama_face_encoding,
    trump_face_encoding
]

known_face_names = [
   "Barack Obama",
   "Donald Trump"
]

#Variable Intialization
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    #Capture Single Video Frame
   ret, frame = vid_cap.read()

    #Process every other frame (save time)
   if process_this_frame:
        #Resizes frame to 1/4 size; faster processing
       small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        #Converts image from BGR(OpenCV) to RGB (face_recognition)
       rgb_small_frame = small_frame[:, :, ::-1]

        #Finds faces and face encodings in current frame
       face_locations = face_recognition.face_locations(rgb_small_frame)
       face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

       face_names = []
       for face_encoding in face_encodings:
            #matches face to known faces
           matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
           name = "Unknown"

           face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
           best_match_index = np.argmin(face_distances)
           if matches[best_match_index]:
               name = known_face_names[best_match_index]
            
           face_names.append(name)

   process_this_frame = not process_this_frame

    #Display Results
   for (top, right, bottom, left), name in zip(face_locations, face_names):
        #Rescale face locations(out of 1/4)
       top *= 4
       right *= 4
       bottom *= 4
       left *= 4

        #Draw box around face
       cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #Draw a label with name
       cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
       font = cv2.FONT_HERSHEY_DUPLEX
       cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    #Display Resulting Image
   cv2.imshow('Video', frame)

    #Hit "q" on keyboard to quit!
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break

#Release handle to the webcam
vid_cap.release()
cv2.destroyAllWindows()

