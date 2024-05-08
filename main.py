from utils import (read_video, 
                   save_video)

from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
def main():
    # Read video
    input_video_path = 'input_videos\input_video.mp4'
    video_frames = read_video(input_video_path)

    # Detect players and ball
    player_tracker = PlayerTracker(model_path='models\yolov8x')
    ball_tracker = BallTracker(model_path='models\yolov5_last.pt')

    player_detectains = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/player_detection.pkl')
    
    ball_detection = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/ball_detection.pkl')
    ball_detection = ball_tracker.interpolate_ball_positions(ball_detection)



    # court line detector model
    court_model_path = 'models\keypoint_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players close to the court
    player_detectains = player_tracker.choose_and_filter_players(court_keypoints, player_detectains)


    # Minicourt
    Mini_court  = MiniCourt(video_frames[0])

    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detection)
    print(f'Ball shot frames: {ball_shot_frames}')

    # Draw output
    ## Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detectains)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detection)

    ## Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = Mini_court.draw_mini_court(output_video_frames)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f'Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, 'output_video/output.avi')




if __name__ == '__main__':
    main()
