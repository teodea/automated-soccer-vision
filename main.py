from utils import read_video, save_video
from trackers import Tracker
import cv2 # Used only for saving cropped images 
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator

def main():
    # Read video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize tracker
    tracker = Tracker('models/yolo11x_epochs100_batch8/best.pt') # Choose model
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False, # Choose to read or write stub
                                       stub_path='stubs/track_stubs_yolo11_final.pkl') # Choose stub path to read or write
    
    # Get object positions
    tracker.add_position_to_tracks(tracks)

    """ # Print number of objects detected at each frame
    for frame_num in range(len(tracks['players'])):
        num_players = len(tracks['players'][frame_num])
        num_referees = len(tracks['referees'][frame_num])
        num_ball = len(tracks['ball'][frame_num]) if 'ball' in tracks else 0

        print(f"Frame {frame_num}: "
            f"Players = {num_players}, "
            f"Referees = {num_referees}, "
            f"Ball = {num_ball}")
    """
        
    """ # Save cropped images of players and referees
    for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]

        # Crop bbox from frame
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Save cropped image
        cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)

        break
    """

    # Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    tracker.add_position_to_tracks({'ball': tracks['ball']})

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True, # Choose to read or write stub
                                                                              stub_path='stubs/camera_movement_stub.pkl') # Choose stub path to read or write
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Assign player team colors
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_dict in enumerate(tracks['players']):
        for player_id, track_info in player_dict.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track_info['bbox'], player_id)
            track_info['team'] = team
            track_info['team_color'] = team_assigner.team_colors[team]

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign ball to player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player is not None:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control != []:
                team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/yolo11_z_final.avi') # Choose output video path

if __name__ == "__main__":
    main()