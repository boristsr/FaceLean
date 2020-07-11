import cv2
import numpy as np
import dlib
import math

from key_events import key_down, key_up

ACTIVATION_THRESHOLD = 0.3
ESC_KEY_CODE = 27
# Colors in BGR format
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)

def get_normalized_vector(vector):
    return vector / np.linalg.norm(vector)

# shape_to_normal and get_eyes_noes_dlib are from 
# https://towardsdatascience.com/precise-face-alignment-with-opencv-dlib-e6c8acead262
def shape_to_normal(shape):
    shape_normal = []
    for i in range(0, 5):
        shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
    return shape_normal

def get_eyes_nose_dlib(shape):
    nose = shape[4][1]
    left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
    left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
    right_eyes_x = int(shape[1][1][0] + shape[0][1][0]) // 2
    right_eyes_y = int(shape[1][1][1] + shape[0][1][1]) // 2
    return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)

left_pressed = False
right_pressed = False

def lean_left_down():
    global left_pressed
    if left_pressed is False:
        key_down('q')
        left_pressed = True
        print("Lean left down")

def lean_left_up():
    global left_pressed
    if left_pressed:
        key_up('q')
        left_pressed = False
        print("Lean left up")

def lean_right_down():
    global right_pressed
    if right_pressed is False:
        key_down('e')
        right_pressed = True
        print("Lean right down")

def lean_right_up():
    global right_pressed
    if right_pressed:
        key_up('e')
        right_pressed = False
        print("Lean right up")

def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

    cap = cv2.VideoCapture(0)

    current_angle = 0.0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_detected = False
        current_dot_product = 0

        rects = detector(gray, 0)
        if len(rects) > 0:
            face_detected = True
            for rect in rects:
                x = rect.left()
                y = rect.top()
                w = rect.right()
                h = rect.bottom()
                shape = predictor(gray, rect)

                shape = shape_to_normal(shape)
                nose, left_eye_tuple, right_eye_tuple = get_eyes_nose_dlib(shape)
                left_eye = np.array(left_eye_tuple)
                right_eye = np.array(right_eye_tuple)

                #Draw the detected eyes and eyeline
                frame = cv2.circle(frame, left_eye_tuple, 5, YELLOW)
                frame = cv2.circle(frame, right_eye_tuple, 5, YELLOW)
                frame = cv2.line(frame, left_eye_tuple, right_eye_tuple, YELLOW)

                #calculate the difference vector between the eyes
                eye_line_vec = right_eye - left_eye
                eye_line_vec = eye_line_vec / 2

                #Draw the center of the eyes
                eye_line_center = left_eye + eye_line_vec
                eye_line_center_tuple = (int(eye_line_center[0]), int(eye_line_center[1]))
                frame = cv2.circle(frame, eye_line_center_tuple, 5, BLUE)

                #draw the level line
                eye_distance = np.linalg.norm(eye_line_vec)
                level_line_1 = np.add(eye_line_center_tuple, (int(-eye_distance), 0))
                level_line_2 = np.add(eye_line_center_tuple, (int(eye_distance), 0))
                frame = cv2.line(frame, tuple(level_line_1), tuple(level_line_2), BLUE)

                eye_line_vec_norm = get_normalized_vector(eye_line_vec)
                current_dot_product = np.dot(eye_line_vec_norm, (0, 1))

        if face_detected:
            if abs(current_dot_product) > ACTIVATION_THRESHOLD:
                #we are ready to activate!
                if current_dot_product > 0.0:
                    lean_left_down()
                    lean_right_up()
                else:
                    lean_right_down()
                    lean_left_up()
            else:
                lean_left_up()
                lean_right_up()

        face_detected_string = f'face detected: {str(face_detected)}'
        angle_string = f'current angle: {str(current_dot_product)}'
        cv2.putText(frame, face_detected_string,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, angle_string,(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0), 2, cv2.LINE_AA)
        # Display the frame with drawn markers
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == ESC_KEY_CODE:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()