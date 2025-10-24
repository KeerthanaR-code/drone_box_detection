from ultralytics import YOLO
cnt = 1
model = YOLO("/home/keerthu/Development/drone_box_detection/best.pt")
results = model(["/home/keerthu/Development/drone_box_detection/SITL_images/img1.png", "/home/keerthu/Development/drone_box_detection/SITL_images/img2.png"], device="cpu")
for result in results:  
    xywh = result.boxes.xywh
    print(f"Image {cnt} xywh: {xywh}")
    img_name = f"result_{cnt}.jpg"
    result.save(filename=img_name)
    cnt+=1