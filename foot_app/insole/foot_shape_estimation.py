from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class FootShapeExtractor:
    def __init__(self,model_path = 'foot_app/insole/models/best.pt' ):
        # Load a model
        self.shape_model = YOLO(model_path)  # pretrained YOLOv8n model

    def predict_shape_mask(self,imag):
        results = self.shape_model.predict(imag)
        for r in results:
            img = np.copy(r.orig_img)
            img_name = Path(r.path).stem
            # iterate each object contour 
            for ci,c in enumerate(r):
                label = c.names[c.boxes.cls.tolist().pop()]

                b_mask = np.zeros(img.shape[:2], np.uint8)
                # Create contour mask 
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
                isolated = cv2.bitwise_and(mask3ch, img)
        #cv2.imshow("resultmask",isolated)
        return mask3ch
    

# # Run batched inference on a list of images
# results = model(['frame_0395.jpg', 'frame_0539.jpg','frame_0995.jpg'])  # return a list of Results objects

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen
#     result.save(filename='result.jpg')  # save to disk