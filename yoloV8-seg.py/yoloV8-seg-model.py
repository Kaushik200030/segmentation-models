import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# Load a pretrained YOLOv8n-seg Segment model
model = YOLO("models/yolov8n-seg.pt")

# Run inference on an image
results = model("assets/5BC0EF4E-9CBD-47E1-B104-647C5EA37795_1_105_c.jpeg")  # results list

# Visualize results
for r in results:
    masks = r.masks  # Masks object containing the detected instance masks
    orig_img = r.orig_img  # Original image

    # Iterate over each mask in the Masks object
    for i, mask in enumerate(masks):
        # Convert mask to numpy array
        mask_array = mask.data.numpy()
        
        # Ensure mask array has the correct shape
        if len(mask_array.shape) == 3:
            mask_array = mask_array[0]

        # Resize the mask to match the original image dimensions
        mask_resized = cv2.resize(mask_array, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Create a blank canvas with the same shape as the original image
        blank_canvas = np.zeros_like(orig_img)

        # Apply the mask to the canvas (mask_resized should be binary)
        blank_canvas[mask_resized == 1] = [255, 0, 0]  # Color the mask area red
        
        # Overlay the mask on the original image
        img_with_mask = cv2.addWeighted(orig_img, 1, blank_canvas, 0.5, 0)
        
        # Display the image with mask
        plt.imshow(cv2.cvtColor(img_with_mask, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Turn off axis labels
        plt.show()

        output_path = f"output_mask_{i}.jpg"
        cv2.imwrite(output_path, img_with_mask)
        print(f"Saved image with mask overlay as {output_path}")
