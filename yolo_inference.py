from ultralytics import YOLO

model = YOLO('models/yolo11x_epochs100_batch8/best.pt')

results = model.predict('input_videos/08fd33_4.mp4', save=True)

print(results)
print('\n\n==================================\n\n')
print(results[0])
print('\n\n==================================\n\n')
i = 1
for box in results[0].boxes:
    print(f'box {i}:')
    print(box)
    print()
    i += 1