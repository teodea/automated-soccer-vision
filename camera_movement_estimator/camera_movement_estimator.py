import pickle
import cv2
import numpy as np
import sys
sys.path.append('../')
from utils import measure_distance, measure_distance_xy
import os

class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5

        first_frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grey)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=3,
            mask=mask_features,
        )

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as stub_file:
                camera_movement = pickle.load(stub_file)
                return camera_movement

        camera_movement = [[0,0]]*len(frames)

        previous_frame_grey = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        previous_frame_features = cv2.goodFeaturesToTrack(previous_frame_grey, **self.features)

        for frame_num in range(1, len(frames)):
            current_frame_grey = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            current_frame_features, status, error = cv2.calcOpticalFlowPyrLK(previous_frame_grey, current_frame_grey, previous_frame_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (current, previous) in enumerate(zip(current_frame_features, previous_frame_features)):
                current_features_flattened = current.ravel()
                previous_features_flattened = previous.ravel()

                distance = measure_distance(current_features_flattened, previous_features_flattened)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_distance_xy(current_features_flattened, previous_features_flattened)

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                previous_frame_features = cv2.goodFeaturesToTrack(current_frame_grey, **self.features)

            previous_frame_grey = current_frame_grey.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as stub_file:
                pickle.dump(camera_movement, stub_file)

        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement: X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement: Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames
    
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    if 'position' not in track_info:
                        print(f"Missing 'position' in {object}, frame {frame_num}, id {track_id}")
                        continue
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0], position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted