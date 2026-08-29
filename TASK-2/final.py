import cv2 
import numpy as np 
import matplotlib.pyplot as plt  

def make_coordinates(lane_image, line_parameters):     
    slope, intercept = line_parameters     
    y1 = lane_image.shape[0]     
    y2 = int(y1 * (3/5))     
    x1 = int((y1 - intercept) / slope)     
    x2 = int((y2 - intercept) / slope)     
    return np.array([x1, y1, x2, y2])  

def canny(lane_image):     
    gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (5, 5), 0)     
    canny = cv2.Canny(blur, 50, 150)     
    return canny  

def region_of_interest(canny):     
    height, width = canny.shape[:2]
    polygons = np.array([[
       ((int(width * 0.05), int(height * 0.95)),(int(width * 0.30), int(height * 0.45)),(int(width * 0.70), int(height * 0.45)) ,(int(width * 0.95), int(height * 0.95)))]], dtype=np.int32)
    
    mask = np.zeros_like(canny)     
    cv2.fillPoly(mask, polygons, 255)  
    cv2.imshow("the mask",mask)
    masked_image = cv2.bitwise_and(canny, mask)     
    return masked_image  

def display_lines(lane_image, lines_dict):     
    line_image = np.zeros_like(lane_image)     
    if lines_dict is not None:
        for line in lines_dict.values():   # Iterates over available left/right lines       
            if line is not None:
                x1, y1, x2, y2 = line.reshape(4)         
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)     
    return line_image  

def average_slope_intercept(lane_image, lines):     
    left_fit = []     
    right_fit = []     
    
    if lines is None:
        return None

    for line in lines:         
        x1, y1, x2, y2 = line.reshape(4)         
        parameters = np.polyfit((x1, x2), (y1, y2), deg=1)         
        slope = parameters[0]         
        intercept = parameters[1]               
        
        # Eliminate nearly horizontal lines that disrupt lane tracking
        if abs(slope) < 0.1: 
            continue
            
        if slope < 0:             
            left_fit.append((slope, intercept))         
        else:             
            right_fit.append((slope, intercept))         
                 
    # Using a dictionary preserves strict identification of which side is which
    lane_lines = {'left': None, 'right': None}
    
    if len(left_fit) > 0:
        left_fit_average = np.median(left_fit, axis=0)         
        lane_lines['left'] = make_coordinates(lane_image, left_fit_average)         

    if len(right_fit) > 0:
        right_fit_average = np.median(right_fit, axis=0)              
        lane_lines['right'] = make_coordinates(lane_image, right_fit_average)         
        
    return lane_lines

# Read input image
image = cv2.imread(r"ugv_r3_task2\10.jpeg")  # Ensure the path is correct and the image exists
if image is None:
    raise FileNotFoundError("Check your image file path. OpenCV could not load it.")

lane_image = np.copy(image)  

# Image Processing Pipeline
canny_image = canny(lane_image) 
masked_image = region_of_interest(canny_image) 

# Hough Transform 
lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)   
averaged_lines = average_slope_intercept(lane_image, lines)  

# Image with lane lines
line_image = display_lines(lane_image, averaged_lines)  
cv2.imshow("lineimg",line_image)
# Lane Overlay
lane_overlay = np.zeros_like(lane_image)

# Verify both keys contain actual valid lane coordinates
if averaged_lines is not None and averaged_lines['left'] is not None and averaged_lines['right'] is not None:
    left_x1, left_y1, left_x2, left_y2 = averaged_lines['left']
    right_x1, right_y1, right_x2, right_y2 = averaged_lines['right']
    
    # Define polygon points for lane area
    lane_points = np.array([[
        [left_x1, left_y1],   # Bottom Left
        [left_x2, left_y2],   # Top Left
        [right_x2, right_y2],  # Top Right
        [right_x1, right_y1]   # Bottom Right
    ]], dtype=np.int32)
    
    cv2.fillPoly(lane_overlay, lane_points, (0, 255, 0))

#blend the lane overlay with the original image
blended_lane = cv2.addWeighted(lane_image, 1.0, lane_overlay, 0.3, 0)

#final combined image
combined_image = cv2.addWeighted(blended_lane, 1.0, line_image, 1.0, 1)   
#combined_image = cv2.bitwise_or(blended_lane, line_image)

cv2.imwrite(r"C:\Users\Dell\Documents\akashsheejith_26_A1_044\TASK-2\Output images\Output-10.png",combined_image)
cv2.imshow("Canny",canny_image)
cv2.imshow("masked",masked_image)
cv2.imshow("Final Output", combined_image) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
