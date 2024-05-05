from ultralytics import YOLO
import cv2
import torch
import pickle

class BallTracker:
    def __init__(self,model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model =YOLO(model_path).to(device)
    
    def detect_frames(self,frames ,read_from_stub=False, stub_path=None): # frames is a list of frames
        ball_detections = [] 

        if read_from_stub and stub_path is not None:  # read detections from stub file
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        if stub_path is not None:  # save detections to stub file
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections


    def detect_frame(self,frame): #
        results = self.model.predict(frame,conf=0.15)[0] # predict the objects in the frame with conf threshold of 0.15

        ball_dict ={}
        for box in results.boxes:
            result =box.xyxy.tolist()[0]
            ball_dict[1] = result


        return ball_dict

    def draw_bboxes(self,video_frames,player_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames,player_detections):
            # draw bboxes on the frame
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f'Ball ID:{track_id}', (int(bbox[0]),int(bbox[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2) # draw track id on the bbox


                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2) # draw bbox with red color for player bbox
            output_video_frames.append(frame)

        


        return output_video_frames