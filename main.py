from utils import (read_video, 
                   save_video)

from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
def main():
    # Read video
    input_video_path = 'input_videos\input_video.mp4'
    video_frames = read_video(input_video_path)

    # Detect players and ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models\yolov5_last.pt')

    player_detectains = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/player_detection.pkl')
    
    ball_detectains = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/ball_detection.pkl')

    # Detect court lines
    

    # Draw output
    ## Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detectains)
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detectains)
    save_video(output_video_frames, 'output_video/output.avi')



if __name__ == '__main__':
    main()
