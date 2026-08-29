import cv2
import numpy as np

filename = r'C:\Users\Dell\Documents\akashsheejith_26_A1_044\TASK-3\ugv_r3_task3\1.png'
image = cv2.imread(filename)

img_h, img_w = image.shape[:2]
output_img = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

color_targets = {
    "Yellow Target": {
        "lower": np.array([20, 100, 50]),
        "upper": np.array([30, 255, 255])
    },
    "Blue Barrel": {
        "lower": np.array([100, 150, 50]),
        "upper": np.array([140, 255, 255])
    },
    "Green Barrel": {
        "lower": np.array([40, 70, 50]),
        "upper": np.array([80, 255, 255])
    }
}

# isolate potholes and lane lines
_, white_mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

for _, bounds in color_targets.items():
    color_mask = cv2.inRange(hsv, bounds["lower"], bounds["upper"])
    color_mask = cv2.dilate(color_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(color_mask))

# isolate potholes
pothole_mask = np.zeros_like(white_mask)
# find contours of the white mask
white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in white_contours:
    area = cv2.contourArea(cnt)
    
    
    if 5 < area < 25000:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Suppress lane boundaries hitting screen edges
        # TWEAK: Added a width check (w > 80) so that long lane lines are skipped, 
        # but small potholes touching lines are allowed to process.
        if (x <= 2 or y <= 2 or (x + w) >= (img_w - 2) or (y + h) >= (img_h - 2)) and w > 80:
            continue  
            
        if len(cnt) >= 5:# if at least 5 points, fit an ellipse to the contour
            _, axes, _ = cv2.fitEllipse(cnt)
            minor_axis = axes[0]
            major_axis = axes[1]
            # if minor axis is greater than 0 then set the value, else set aspect ratio to 0(ternary operator)
            ellipse_aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
            
            # TWEAK: Stretched upper limit to 16.0 to allow for flat, perspective-distorted shapes near the horizon
            if 1.0 <= ellipse_aspect_ratio < 16.0:
                cv2.drawContours(pothole_mask, [cnt], -1, 255, -1)
        else:
            box_aspect_ratio = w / float(h) if h > 0 else 0
            # TWEAK: Expanded constraint down to 0.4 to catch smaller far-away profiles
            if box_aspect_ratio > 0.4: #for when object is far away or too small to fit an ellipse
                cv2.drawContours(pothole_mask, [cnt], -1, 255, -1)

total_hazards = 0

# Process potholes First
pothole_contours, _ = cv2.findContours(pothole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in pothole_contours:
    if cv2.contourArea(contour) > 4:
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + (w // 2)
        center_y = y + (h // 2)
        
        total_hazards += 1
        
        # Output coordinates to console
        print(f"X: {center_x}, Y: {center_y}")
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2, lineType=cv2.LINE_AA)

# Process Colored Obstacles and Barrels
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# from the color_targets dictionary, iterate through each color and its bounds
for color_name, bounds in color_targets.items():
    barrel_color_mask = cv2.inRange(hsv, bounds["lower"], bounds["upper"])
    
    barrel_color_mask = cv2.morphologyEx(barrel_color_mask, cv2.MORPH_OPEN, kernel)
    barrel_color_mask = cv2.morphologyEx(barrel_color_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(barrel_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:
            x, y, w, h = cv2.boundingRect(contour)
            
            if x <= 2 or y <= 2 or (x + w) >= (img_w - 2):
                continue
            center_x = x + (w // 2)
            center_y = y + (h // 2)
            
            total_hazards += 1
            
            print(f"X: {center_x}, Y: {center_y}")
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2, lineType=cv2.LINE_AA)


# final output
cv2.putText(output_img, f"Hazards: {total_hazards}", (30, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, lineType=cv2.LINE_AA)
cv2.imwrite(r"C:\Users\Dell\Documents\akashsheejith_26_A1_044\TASK-3\Output images\Output-1.png",output_img)
cv2.imshow("OUTPUT",output_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
