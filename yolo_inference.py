from ultralytics import YOLO

model = YOLO('models\chick_last.pt')

result=model.track('input_videos/chick.mp4',conf=0.2, save=True)
#print(result)
#print('boxes:')
#for box in result[0].boxes:
#    print(box)