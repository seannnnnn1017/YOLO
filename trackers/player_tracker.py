from ultralytics import YOLO
import cv2
import torch
import pickle
class PlayerTracker:
    def __init__(self,model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model =YOLO(model_path).to(device)
    
    def detect_frames(self,frames ,read_from_stub=False, stub_path=None): # frames is a list of frames
        player_detections = [] 

        if read_from_stub and stub_path is not None:  # read detections from stub file
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:  # save detections to stub file
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections


    def detect_frame(self,frame): #
        results = self.model.track(frame, persist=True)[0] # return a list of detections with track id and bbox coordinates     #persist=True 表示追?信息會在多個幀之間持續保存
        id_name_dict = results.names # a dictionary of id and name of objects in the model  

        player_dict ={}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result =box.xyxy.tolist()[0] #獲取邊界框的座標，通常為 [x_min, y_min, x_max, y_max]
            object_cls_id =box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == 'person':
                player_dict[track_id] = result


        return player_dict

    def draw_bboxes(self,video_frames,player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames,player_detections):
            # draw bboxes on the frame
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f'player ID:{track_id}', (int(bbox[0]),int(bbox[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) # draw track id on the bbox


                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2) # draw bbox with red color for player bbox
            output_video_frames.append(frame)

        


        return output_video_frames