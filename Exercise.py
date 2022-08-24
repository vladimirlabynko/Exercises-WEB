
import mediapipe as mp
import cv2

import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


def convert_to_point(landmark, image_width, image_height):
    return tuple(np.multiply(landmark, [image_width, image_height]).astype(int))


def render_extremities(image, color, point_1, point_2):
    cv2.line(image, point_1, point_2, color, thickness=10, lineType=8)


def render_point(image, color1, color2, point):
    cv2.circle(image, point, 10, color1, cv2.FILLED)
    cv2.circle(image, point, 12, color2, 2)


def correct_toe_left(foot, knee, image_height, image_width):
    correct = None
    a = np.multiply(foot, [image_width, image_height]).astype(int)
    b = np.multiply(knee, [image_width, image_height]).astype(int)
    if a[0] > b[0]:
        correct = "BAD"
    else:
        correct = "GOOD"

    return correct


def correct_toe_right(foot, knee, image_height, image_width):
    correct = None
    a = np.multiply(foot, [image_width, image_height]).astype(int)
    b = np.multiply(knee, [image_width, image_height]).astype(int)
    if a[0] > b[0]:
        correct = "GOOD"
    else:
        correct = "BAD"

    return correct


def true_position_down_right(image, color, point):
    one = list(point)

    one[0] += 200
    two = list(point)
    two[0] += 200
    two[1] += 200
    cv2.line(image, point, one, color, thickness=4, lineType=8)
    cv2.line(image, one, two, color, thickness=4, lineType=8)


def true_position_up_right(image, color, point):
    one = list(point)
    one[1] += 200

    two = list(point)

    two[0] -= 200
    two[1] += 200
    cv2.line(image, point, one, color, thickness=4, lineType=8)
    cv2.line(image, one, two, color, thickness=4, lineType=8)


def true_position_down_left(image, color, point):
    one = list(point)
    one[1] += 200

    two = list(point)

    two[0] += 200
    two[1] += 200
    cv2.line(image, point, one, color, thickness=4, lineType=8)
    cv2.line(image, one, two, color, thickness=4, lineType=8)


def true_position_up_left(image, color, point):
    one = list(point)
    one[0] -= 200

    two = list(point)

    two[0] -= 200
    two[1] += 200
    cv2.line(image, point, one, color, thickness=4, lineType=8)
    cv2.line(image, one, two, color, thickness=4, lineType=8)


GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
ORANGE = (0, 165, 255)


def good_morning():
    cap = cv2.VideoCapture(0)
# Setup mediapipe instance
    if (cap.isOpened() == False):
        print("Error opening the video file")
# Obtain frame size information using get() method
    image_width = int(cap.get(3))
    image_height = int(cap.get(4))
    upper_left = (image_width // 10, image_height // 10)
    bottom_right = (image_width * 9 // 10, image_height * 9 // 10)
    stage = "Start"
    stage_info = (image_width // 4, image_height // 10)
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
        # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

        # Make detection
            results = pose.process(image)

        # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

            # Get coordinates
                knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                l_knee = convert_to_point(knee_left, image_width, image_height)
                r_knee = convert_to_point(
                    knee_right, image_width, image_height)

                l_ankle = convert_to_point(
                    ankle_left, image_width, image_height)
                r_ankle = convert_to_point(
                    ankle_right, image_width, image_height)

                l_hip = convert_to_point(hip_left, image_width, image_height)
                r_hip = convert_to_point(hip_right, image_width, image_height)

                l_shoulder = convert_to_point(
                    left_shoulder, image_width, image_height)
                r_shoulder = convert_to_point(
                    right_shoulder, image_width, image_height)

                l_elbow = convert_to_point(
                    left_elbow, image_width, image_height)
                r_elbow = convert_to_point(
                    right_elbow, image_width, image_height)

                l_wrist = convert_to_point(
                    left_wrist, image_width, image_height)
                r_wrist = convert_to_point(
                    right_wrist, image_width, image_height)

                angle_elbow_left = round(calculate_angle(
                    left_wrist, left_elbow, left_shoulder), 2)
                angle_elbow_right = round(calculate_angle(
                    right_wrist, right_elbow, right_shoulder), 2)

                angle_back_right = round(calculate_angle(
                    right_shoulder, hip_right, knee_right), 2)

                angle_leg_left = round(calculate_angle(
                    hip_left, knee_left, ankle_left), 2)
                angle_leg_right = round(calculate_angle(
                    hip_right, knee_right, ankle_right), 2)

                render_point(image, BLACK, RED, l_ankle)
                render_point(image, BLACK, RED, l_knee)
                render_point(image, BLACK, RED, l_hip)
                render_point(image, BLACK, RED, l_shoulder)
                render_point(image, BLACK, RED, l_elbow)
                render_point(image, BLACK, RED, l_wrist)

                render_point(image, BLACK, RED, r_ankle)
                render_point(image, BLACK, RED, r_knee)
                render_point(image, BLACK, RED, r_hip)
                render_point(image, BLACK, RED, r_shoulder)
                render_point(image, BLACK, RED, r_elbow)
                render_point(image, BLACK, RED, r_wrist)

                render_extremities(image, WHITE, l_ankle, l_knee)
                render_extremities(image, WHITE, l_knee, l_hip)
                render_extremities(image, WHITE, l_hip, l_shoulder)
                render_extremities(image, WHITE, l_shoulder, l_elbow)
                render_extremities(image, WHITE, l_elbow, l_wrist)

                render_extremities(image, WHITE, r_ankle, r_knee)
                render_extremities(image, WHITE, r_knee, r_hip)
                render_extremities(image, WHITE, r_hip, r_shoulder)
                render_extremities(image, WHITE, r_shoulder, r_elbow)
                render_extremities(image, WHITE, r_elbow, r_wrist)

                # Visualize angle
                cv2.putText(image, str(angle_elbow_left),
                            tuple(np.multiply(left_elbow, [
                                  image_width, image_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2, cv2.LINE_AA)
                cv2.putText(image, str(angle_elbow_right),
                            tuple(np.multiply(right_elbow, [
                                  image_width, image_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2, cv2.LINE_AA)

                cv2.putText(image, str(angle_back_right),
                            tuple(np.multiply(hip_right, [
                                  image_width, image_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2, cv2.LINE_AA)

                cv2.putText(image, str(angle_leg_left),
                            tuple(np.multiply(knee_left, [
                                  image_width, image_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)
                cv2.putText(image, str(angle_leg_right),
                            tuple(np.multiply(knee_right, [
                                  image_width, image_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)


                if angle_elbow_left >= 40 and angle_elbow_right >= 40 and angle_elbow_left <= 80 and angle_elbow_right <= 80:
                    render_extremities(image, GREEN, l_shoulder, l_elbow)
                    render_extremities(image, GREEN, l_elbow, l_wrist)
                    render_extremities(image, GREEN, r_shoulder, r_elbow)
                    render_extremities(image, GREEN, r_elbow, r_wrist)
                    cv2.rectangle(image, upper_left,
                                  bottom_right, GREEN, thickness=5)
                    stage = "KNEE"

                if stage == "KNEE" and angle_leg_right >= 120 and angle_leg_right <= 160:
                    render_extremities(image, GREEN, r_hip, r_knee)
                    render_extremities(image, GREEN, r_knee, r_ankle)
                    render_extremities(image, GREEN, l_hip, l_knee)
                    render_extremities(image, GREEN, l_knee, l_ankle)
                    cv2.rectangle(image, upper_left,
                                  bottom_right, GREEN, thickness=5)
                    stage = "DOWN"

                if stage == "DOWN" and angle_back_right >= 90 and angle_back_right <= 130:
                    render_extremities(image, GREEN, r_shoulder, r_hip)
                    render_extremities(image, GREEN, l_shoulder, l_hip)
                    cv2.rectangle(image, upper_left,
                                  bottom_right, GREEN, thickness=5)
                    stage = "UP"

            except:
                pass

        # Stage data
            cv2.putText(image, str(stage),
                        stage_info,
                        cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def cabaret():
    cap = cv2.VideoCapture(0)
# Setup mediapipe instance
    if (cap.isOpened() == False):
        print("Error opening the video file")
# Obtain frame size information using get() method
    image_width = int(cap.get(3))
    image_height = int(cap.get(4))
    upper_left = (image_width // 10, image_height // 10)
    bottom_right = (image_width * 9 // 10, image_height * 9 // 10)
    stage = "Start"
    counter = 0
    side = None
    stage_info = (image_width // 4, image_height // 10)
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
        # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

        # Make detection
            results = pose.process(image)

        # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

            # Get coordinates
                knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                l_knee = convert_to_point(knee_left, image_width, image_height)
                r_knee = convert_to_point(
                    knee_right, image_width, image_height)

                l_ankle = convert_to_point(
                    ankle_left, image_width, image_height)
                r_ankle = convert_to_point(
                    ankle_right, image_width, image_height)

                l_hip = convert_to_point(hip_left, image_width, image_height)
                r_hip = convert_to_point(hip_right, image_width, image_height)

                l_shoulder = convert_to_point(
                    left_shoulder, image_width, image_height)
                r_shoulder = convert_to_point(
                    right_shoulder, image_width, image_height)

                l_elbow = convert_to_point(
                    left_elbow, image_width, image_height)
                r_elbow = convert_to_point(
                    right_elbow, image_width, image_height)

                l_wrist = convert_to_point(
                    left_wrist, image_width, image_height)
                r_wrist = convert_to_point(
                    right_wrist, image_width, image_height)

                angle_shoulder_left = round(calculate_angle(
                    left_elbow, left_shoulder, hip_left), 2)
                angle_shoulder_right = round(calculate_angle(
                    right_elbow, right_shoulder, hip_right), 2)

                angle_knee_right = round(calculate_angle(
                    hip_right, knee_right, ankle_right), 2)
                angle_knee_left = round(calculate_angle(
                    hip_left, knee_left, ankle_left), 2)

                angle_leg_right = round(calculate_angle(
                    right_shoulder, hip_right, ankle_right), 2)
                angle_leg_left = round(calculate_angle(
                    left_shoulder, hip_left, ankle_left), 2)

                render_point(image, BLACK, RED, l_ankle)
                render_point(image, BLACK, RED, l_knee)
                render_point(image, BLACK, RED, l_hip)
                render_point(image, BLACK, RED, l_shoulder)
                render_point(image, BLACK, RED, l_elbow)
                render_point(image, BLACK, RED, l_wrist)

                render_point(image, BLACK, RED, r_ankle)
                render_point(image, BLACK, RED, r_knee)
                render_point(image, BLACK, RED, r_hip)
                render_point(image, BLACK, RED, r_shoulder)
                render_point(image, BLACK, RED, r_elbow)
                render_point(image, BLACK, RED, r_wrist)

                render_extremities(image, WHITE, l_ankle, l_knee)
                render_extremities(image, WHITE, l_knee, l_hip)
                render_extremities(image, WHITE, l_hip, l_shoulder)
                render_extremities(image, WHITE, l_shoulder, l_elbow)
                render_extremities(image, WHITE, l_elbow, l_wrist)

                render_extremities(image, WHITE, r_ankle, r_knee)
                render_extremities(image, WHITE, r_knee, r_hip)
                render_extremities(image, WHITE, r_hip, r_shoulder)
                render_extremities(image, WHITE, r_shoulder, r_elbow)
                render_extremities(image, WHITE, r_elbow, r_wrist)

                # Visualize angle
                cv2.putText(image, str(angle_shoulder_left),
                            tuple(np.multiply(left_shoulder, [
                                  image_width, image_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)
                cv2.putText(image, str(angle_shoulder_left),
                            tuple(np.multiply(right_shoulder, [
                                  image_width, image_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)

                cv2.putText(image, str(angle_knee_left),
                            tuple(np.multiply(knee_left, [
                                  image_width, image_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2, cv2.LINE_AA)
                cv2.putText(image, str(angle_knee_right),
                            tuple(np.multiply(knee_right, [
                                  image_width, image_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2, cv2.LINE_AA)

                cv2.putText(image, str(angle_leg_right),
                            tuple(np.multiply(ankle_right, [
                                  image_width, image_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)
                cv2.putText(image, str(angle_leg_left),
                            tuple(np.multiply(ankle_left, [
                                  image_width, image_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)


                if (counter % 2) == 0:
                    side = "LEFT"
                    render_extremities(image, YELLOW, l_shoulder, l_elbow)
                    render_extremities(image, YELLOW, l_elbow, l_wrist)
                    render_extremities(image, YELLOW, r_hip, r_knee)
                    render_extremities(image, YELLOW, r_knee, r_ankle)
                else:
                    side = "RIGHT"
                    render_extremities(image, YELLOW, r_shoulder, r_elbow)
                    render_extremities(image, YELLOW, r_elbow, r_wrist)
                    render_extremities(image, YELLOW, l_hip, l_knee)
                    render_extremities(image, YELLOW, l_knee, l_ankle)

                if angle_shoulder_left >= 80 and angle_shoulder_right >= 80:
                    render_extremities(image, GREEN, l_shoulder, l_elbow)
                    render_extremities(image, GREEN, l_elbow, l_wrist)
                    render_extremities(image, GREEN, r_shoulder, r_elbow)
                    render_extremities(image, GREEN, r_elbow, r_wrist)
                    stage = "KNEE"
                else:
                    render_extremities(image, RED, l_shoulder, l_elbow)
                    render_extremities(image, RED, l_elbow, l_wrist)
                    render_extremities(image, RED, r_shoulder, r_elbow)
                    render_extremities(image, RED, r_elbow, r_wrist)

                if side == "LEFT":
                    if stage == "KNEE" and angle_knee_right >= 80 and angle_knee_right <= 100:
                        stage = "LEG"
                        render_extremities(image, GREEN, r_hip, r_knee)

                    if stage == "LEG" and angle_leg_right >= 90 and angle_leg_right <= 110:
                        stage = "STAND"
                        counter += 1
                        render_extremities(image, GREEN, r_knee, r_ankle)

                if side == "RIGHT":
                    if stage == "KNEE" and angle_knee_left >= 80 and angle_knee_left <= 100:
                        stage = "LEG"
                        render_extremities(image, GREEN, r_hip, r_knee)

                    if stage == "LEG" and angle_leg_left >= 90 and angle_leg_left <= 110:
                        stage = "STAND"
                        counter += 1
                        render_extremities(image, GREEN, r_knee, r_ankle)

            except:
                pass

        # Stage data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)

            cv2.putText(image, 'FEET', (125, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
            cv2.putText(image, str(side),
                        (100, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)

            cv2.putText(image, str(stage),
                        stage_info,
                        cv2.FONT_HERSHEY_SIMPLEX, 2, BLACK, 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
