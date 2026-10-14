import argparse
import os

import numpy as np

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator

# Manual corrections for the sample clip (input_videos/08fd33_4.mp4).
# The track IDs are only valid for the tracks stored in stubs/track_stubs_yolo11_final.pkl;
# on another video, or after re-running detection, they have to be found again (or left empty).
SAMPLE_CLIP_REFEREE_TRACK_IDS = [17, 52, 96, 100, 226]  # referees the detector labels as players
SAMPLE_CLIP_FIXED_TEAM_IDS = {87: 1, 115: 1}  # goalkeeper, whose kit colour does not match his team


def parse_args():
    parser = argparse.ArgumentParser(description='Annotate a broadcast football clip with tracks, teams, possession, speed and distance.')
    parser.add_argument('--input', default='input_videos/08fd33_4.mp4', help='input video')
    parser.add_argument('--output', default='output_videos/output.avi', help='output video (XVID .avi)')
    parser.add_argument('--model', default='models/best.pt', help='YOLO weights (see README for the download)')
    parser.add_argument('--use-stubs', action='store_true',
                        help='read cached tracks and camera movement from stubs/ instead of running detection (no weights needed)')
    parser.add_argument('--track-stub', default='stubs/track_stubs_yolo11_final.pkl')
    parser.add_argument('--camera-stub', default='stubs/camera_movement_stub.pkl')
    args = parser.parse_args()
    if not args.use_stubs:
        # a live run would otherwise overwrite the cached stubs
        args.track_stub = args.camera_stub = None
    return args


def main():
    args = parse_args()

    # Read video
    video_frames = read_video(args.input)

    # Detect and track players, referees and ball
    tracker = Tracker(args.model, referee_track_ids=SAMPLE_CLIP_REFEREE_TRACK_IDS)
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=args.use_stubs,
                                       stub_path=args.track_stub)

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    tracker.add_position_to_tracks({'ball': tracks['ball']})

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=args.use_stubs,
                                                                              stub_path=args.camera_stub)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Assign player team colors
    team_assigner = TeamAssigner(fixed_team_ids=SAMPLE_CLIP_FIXED_TEAM_IDS)
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
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    save_video(output_video_frames, args.output)


if __name__ == "__main__":
    main()
