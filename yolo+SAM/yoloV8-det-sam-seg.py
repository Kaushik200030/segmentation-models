from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
import pickle
import cv2
import numpy as np
import time

ISDEBUG = False


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))



image_path = '/Users/kaushikpattanayak/Desktop/KP_Flam/SEGMENTATION/segmentation models/assets/6A92DD4D-200B-4E30-951E-17E1078E8EE6_1_105_c.jpeg'
image  = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model = YOLO("models/yolov8n.pt")
objects = model(image, classes= [0])

for results in objects:
    boxes = results.boxes
    cls = boxes.cls
    classname = "Person"
    xyxy = boxes.xyxy

    if(ISDEBUG):
        print(f"box output :{boxes}")
        print(f"xyxy :{xyxy}")


    if len(cls)>0 and cls[0] == 0.0:
 
        x1 = int(xyxy[0][0])
        y1 = int(xyxy[0][1])
        x2 = int(xyxy[0][2]) 
        y2 = int(xyxy[0][3]) 

        width = abs(x1-x2)
        height = abs(y1-y2)

        x1 = int(x1- (width*0.1))
        y1 = int(y1- (height*0.1))   
        x2 = int(x2+ (width*0.1))
        y2 = int(y2+ (height*0.1))    

        mid_x = int((x1+x2)/2)
        mid_y = int((y1+y2)/2)

        box = np.array([x1,y1,x2,y2])
        point_coords = np.array([[mid_x,mid_y]])
        point_labels = np.array([1])

        if(ISDEBUG):
            print(f"box cordinates: {x1,y1,x2,y2}")


        ##Draw Rect on the image
            
        # cv2.rectangle(image, (x1,y1),(x2,y2), color=(0,255,0),thickness = 2)
        # text = classname
        # font = cv2.FONT_HERSHEY_COMPLEX
        # font_scale = 1.5
        # thickness = 4
        # text_size, _ = cv2.getTextSize(text,font,font_scale,thickness=thickness)
        # text_x = int(x1+5)
        # text_y = int(y1 +text_size[1]+5)
        # cv2.putText(image,text,(text_x,text_y), font, font_scale,thickness)

        ####################################################################################
        # SAM Model starts here
        ####################################################################################
        import sys
        sys.path.append("..")
        from segment_anything import sam_model_registry, SamPredictor

        sam_checkpoint = "/Users/kaushikpattanayak/Desktop/KP_Flam/SEGMENTATION/segmentation models/yolo+SAM/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "mps"

        sam = sam_model_registry[model_type](checkpoint= sam_checkpoint)
        sam.to(device= device)

        predictor = SamPredictor(sam)
        predictor.set_image(image)

        in_time = time.time()
        masks,_,_ = predictor.predict(
            point_coords = None,
            point_labels = None,
            box = box,
            multimask_output= False
        )
        out_time = time.time()

        inference_time = out_time - in_time
        print(f"inference time:{inference_time}")

        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(masks[0], plt.gca())
        show_box(box,plt.gca())
        plt.axis('off')
        random_number = np.random.randint(1, 1001)
        plt.savefig(f'output_sam_seg{random_number}.png')
        plt.show()  




