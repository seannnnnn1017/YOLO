from ultralytics import YOLO
import cv2
import torch
import pickle
import pandas as pd


class BallTracker:
    def __init__(self,model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model =YOLO(model_path).to(device)
    
    def interpolate_ball_positions(self, ball_positions): 
        # ball_positions = [{}, {1: [892.9063720703125, 616.3403930664062, 911.8057861328125, 635.3505859375]}, {}, {}, {}] # example ball positions
                               #{id: [x1,y1,x2,y2]}

        ball_positions =[x.get(1,[]) for x in ball_positions]
        # covert the list of lists to a dataframe
        df_ball_positions =pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values in the dataframe
        df_ball_positions = df_ball_positions.interpolate(method='linear', axis=0)
        df_ball_positions = df_ball_positions.bfill() # backfill the missing values

        ball_positions =[{1:x} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def get_ball_shot_frames(self,ball_positions):
        ball_positions =[x.get(1,[]) for x in ball_positions]
        # covert the list of lists to a dataframe
        df_ball_positions =pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        #�p�� mid_y_rolling_mean ��쪺�۾F�����������t���A�ñN���G�s�x��s����� delta_y ���C
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        df_ball_positions['ball_hit']=0
        minimum_change_frames_for_hit = 25  # �w�q�P�_�y�Q�����һݪ��̤p�ܤƴV��

        # �M���C�@�ӴV�A�ˬd�y�O�_�Q����
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            # �ˬd�۾F�V������m�ܤƬO�_�ŦX�y�Q���������p
            negative_positions_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0
            positive_positions_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0
            
            # �p�G�o�{�ŦX���������ܤơA�i�@�B�ˬd���U�Ӫ��X�ӴV�O�_�٦s�b�ۦP���ܤƼҦ�
            if negative_positions_change or positive_positions_change:
                
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    # �ˬd���U�Ӫ��X�ӴV����m�ܤƬO�_�ŦX�y�Q���������p�A�òέp�ܤƪ��V��
                    negative_positions_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_positions_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_positions_change and negative_positions_change_following_frame:
                        change_count += 1
                    elif positive_positions_change and positive_positions_change_following_frame:
                        change_count += 1

                # �p�G�s��V�ƶW�L�F minimum_change_frames_for_hit - 1�A�h�аO�y�Q����
                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions['ball_hit'].loc[i] = 1
        frame_nums_with_ball_hit = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()
        return frame_nums_with_ball_hit

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

    def draw_bboxes(self,video_frames,ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames,ball_detections):
            # draw bboxes on the frame
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f'Ball ID:{track_id}', (int(bbox[0]),int(bbox[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2) # draw track id on the bbox


                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2) # draw bbox with red color for player bbox
            output_video_frames.append(frame)

        


        return output_video_frames