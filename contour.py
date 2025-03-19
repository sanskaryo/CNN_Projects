import cv2
import numpy as np
import os

def detect_nail_contour(image_path, output_txt="contour_coordinates.txt", save_image=True, display_result=True):

    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return False

    orig = img.copy()
    orig_h, orig_w = orig.shape[:2]

    proc_width = 500
    if orig_w > proc_width:
        scale_factor = proc_width / orig_w
        img = cv2.resize(img, (proc_width, int(orig_h * scale_factor)))
    else:
        scale_factor = 1.0

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  
    _, a_channel, _ = cv2.split(lab)

    _, thresh = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

  
    contours_info = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    if not contours:
        print("No contours detected.")
        return False

    nail_contour = max(contours, key=cv2.contourArea)

 
    if scale_factor != 1.0:
        scale = np.array([1/scale_factor, 1/scale_factor])
        nail_contour = (nail_contour * scale).astype(np.int32)

  
    epsilon = 0.005 * cv2.arcLength(nail_contour, True)
    approx = cv2.approxPolyDP(nail_contour, epsilon, True)

    save_contour_coordinates(approx, orig.shape, output_txt)

   
    result_img = orig.copy()
    cv2.drawContours(result_img, [approx], -1, (0, 255, 0), 2)


    if save_image:
        output_image_path = generate_output_image_path(image_path)
        cv2.imwrite(output_image_path, result_img)
        print(f"Contour image saved as: {output_image_path}")

    if display_result:
        cv2.imshow("Nail Contour Detection", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Contour coordinates saved to {output_txt}")
    return True

def save_contour_coordinates(contour, img_shape, filename):
    """
    Saves the contour coordinates to a text file.
    Each line contains x,y coordinates of one point.
    """
    with open(filename, 'w') as f:
        f.write(f"Image Dimensions: {img_shape[1]}x{img_shape[0]}\n")
        f.write("Contour Coordinates (x,y):\n")
      
        points = contour.squeeze()
        if points.ndim == 1:
            points = np.expand_dims(points, axis=0)
        for point in points:
            f.write(f"{int(point[0])},{int(point[1])}\n")

def generate_output_image_path(image_path):
    base, ext = os.path.splitext(image_path)
    return f"{base}_contour{ext}"

if __name__ == "__main__":
    # Example usage: Provide top view or front view finger image.
    input_image = "nail2.jpeg"
    detect_nail_contour(input_image)


# import cv2
# import numpy as np

# def remove_green(img):
#     empty_img = np.zeros_like(img)
#     RED, GREEN, BLUE = (2, 1, 0)
#     reds = img[:, :, RED]
#     greens = img[:, :, GREEN]
#     blues = img[:, :, BLUE]
#     # loop over the image, pixel by pixel
#     tmpMask = (greens < 35) | (reds > greens) | (blues > greens)
#     img[tmpMask == 0] = (0, 0, 0)  # remove background from original picture
#     empty_img[tmpMask] = (255, 255, 255)  # mask with finger in white
#     return img, empty_img

# def detect_nail(gray_mask):
#     # Find contours in the mask
#     contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Find the largest contour which should be the finger
#     if len(contours) > 0:
#         largest_contour = max(contours, key=cv2.contourArea)
        
#         # Get bounding rectangle
#         x, y, w, h = cv2.boundingRect(largest_contour)
        
#         # Assume nail is in upper portion of finger
#         nail_region = gray_mask[y:y+int(h*0.4), x:x+w]
        
#         return nail_region, (x, y, w, h)
#     return None, None

# def main():
#     # Define imagePath before using it
#     imagePath = r"C:\Sankhu Codes and Stuff\Machine learning and related\DL PROJECT\nail2.jpeg"  # Replace with actual image path
    
#     # load and process 
#     image = cv2.imread(imagePath, 1)  # load
#     image = cv2.resize(image, None, fx=0.3, fy=0.3)  # resize
#     image = cv2.GaussianBlur(image, (3, 3), 0)
#     no_green_image, mask_finger = remove_green(image)  # remove green
#     gray = cv2.cvtColor(no_green_image, cv2.COLOR_BGR2GRAY)  # gray scaled
#     gray_mask_finger = cv2.cvtColor(mask_finger, cv2.COLOR_BGR2GRAY)

#     # refine edges
#     kernel = np.ones((5, 5), np.uint8)
#     gray_mask_finger = cv2.morphologyEx(gray_mask_finger, cv2.MORPH_GRADIENT, kernel)

#     nail_region, bbox = detect_nail(gray_mask_finger)
    
#     if nail_region is not None:
#         # Draw bounding box on original image
#         x, y, w, h = bbox
#         cv2.rectangle(image, (x, y), (x+w, y+int(h*0.4)), (0, 255, 0), 2)
        
#         # Display results
#         cv2.imshow('Original with nail detection', image)
#         cv2.imshow('Nail region', nail_region)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

# def remove_green(img):
#     empty_img = np.zeros_like(img)
#     RED, GREEN, BLUE = (2, 1, 0)
#     reds = img[:, :, RED]
#     greens = img[:, :, GREEN]
#     blues = img[:, :, BLUE]
#     # loop over the image, pixel by pixel
#     tmpMask = (greens < 35) | (reds > greens) | (blues > greens)
#     img[tmpMask == 0] = (0, 0, 0)  # remove background from original picture
#     empty_img[tmpMask] = (255, 255, 255)  # mask with finger in white
#     return img, empty_img

# # main function
# # load and process 
# image = cv2.imread(imagePath, 1)  # load
# image = cv2.resize(image, None, fx=0.3, fy=0.3)  # resize
# image = cv2.GaussianBlur(image, (3, 3), 0)
# no_green_image, mask_finger = remove_green(image)  # remove green
# gray = cv2.cvtColor(no_green_image, cv2.COLOR_BGR2GRAY)  # gray scalEd
# gray_mask_finger = cv2.cvtColor(mask_finger, cv2.COLOR_BGR2GRAY)

# # refine edges
# kernel = np.ones((5, 5), np.uint8)
# gray_mask_finger = cv2.morphologyEx(gray_mask_finger, cv2.MORPH_GRADIENT, kernel)

# detect_nail(gray_mask_finger)