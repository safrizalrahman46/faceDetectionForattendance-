import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Initialize video capture
video_capture = cv2.VideoCapture(1)

if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

# Load known face images and encodings
known_faces = {
    "faruq": face_recognition.face_encodings(face_recognition.load_image_file("foto/faruq.jpg"))[0],
    "nazril": face_recognition.face_encodings(face_recognition.load_image_file("foto/nazril.jpg"))[0],
    "rahman": face_recognition.face_encodings(face_recognition.load_image_file("foto/rahman.jpg"))[0],
    "rizqi": face_recognition.face_encodings(face_recognition.load_image_file("foto/rizqi.jpg"))[0],
    "satria": face_recognition.face_encodings(face_recognition.load_image_file("foto/satria.jpg"))[0],
}

known_face_encoding = list(known_faces.values())
known_faces_names = list(known_faces.keys())

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

# Create a CSV file for attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

with open(f"{current_date}.csv", 'w+', newline='') as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Name", "Time"])

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Process the frame
        if s:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            print("Detected face locations:", face_locations)

            if face_locations:
                try:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                except Exception as e:
                    print("Error encoding face:", e)
                    face_encodings = []  # Skip this frame if encoding fails

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
                name = ""
                if True in matches:
                    best_match_index = np.argmin(face_recognition.face_distance(known_face_encoding, face_encoding))
                    if matches[best_match_index]:
                        name = known_faces_names[best_match_index]

                face_names.append(name)

                if name and name in students:
                    students.remove(name)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])
        else:
            print("No faces detected in this frame.")

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Label the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Show the video frame
        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
