import numpy as np
import cv2


def detect_squares_in_image(rgb_image, target_color=(255, 120, 55), tolerance=5, step=2, aspect_ratio_threshold=1.3,
                            min_aspect_ratio=0.5, overlap_threshold=0.5):
    # Convert the RGB image to BGR format
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Create a mask for non-black pixels
    non_black_mask = np.any(bgr_image != [0, 0, 0], axis=-1)

    horizontal_projection = np.sum(non_black_mask, axis=1)
    vertical_projection = np.sum(non_black_mask, axis=0)

    # Detect gaps in the horizontal and vertical projections (indicating separation between squares)
    horizontal_gaps = np.where(horizontal_projection[::step] == 0)[0]
    vertical_gaps = np.where(vertical_projection[::step] == 0)[0]

    # Identify bounding boxes based on the gaps
    bounding_boxes = []
    height, width, _ = bgr_image.shape

    if len(horizontal_gaps) > 0:
        y_starts = [0] + list(horizontal_gaps * step + step)
        y_ends = list(horizontal_gaps * step) + [height - 1]

        for y_start, y_end in zip(y_starts, y_ends):
            if np.any(horizontal_projection[y_start:y_end] > 0):
                x_min = np.min(np.where(non_black_mask[y_start:y_end, :])[1])
                x_max = np.max(np.where(non_black_mask[y_start:y_end, :])[1])
                bounding_boxes.append((x_min, y_start, x_max, y_end))

    if len(vertical_gaps) > 0:
        x_starts = [0] + list(vertical_gaps * step + step)
        x_ends = list(vertical_gaps * step) + [width - 1]

        for x_start, x_end in zip(x_starts, x_ends):
            if np.any(vertical_projection[x_start:x_end] > 0):
                y_min = np.min(np.where(non_black_mask[:, x_start:x_end])[0])
                y_max = np.max(np.where(non_black_mask[:, x_start:x_end])[0])
                bounding_boxes.append((x_start, y_min, x_end, y_max))

    # Filter bounding boxes based on aspect ratio
    filtered_boxes = []
    for (x_min, y_min, x_max, y_max) in bounding_boxes:
        aspect_ratio = (x_max - x_min) / (y_max - y_min)
        if 1 / aspect_ratio_threshold < aspect_ratio < aspect_ratio_threshold and aspect_ratio > min_aspect_ratio:
            filtered_boxes.append((x_min, y_min, x_max, y_max))

    # Merge overlapping or nearby bounding boxes
    if not filtered_boxes:
        return []

    # Sort the bounding boxes by their starting x-coordinate
    filtered_boxes = sorted(filtered_boxes, key=lambda x: x[0])

    merged_boxes = [filtered_boxes[0]]

    for i in range(1, len(filtered_boxes)):
        x1, y1, x2, y2 = filtered_boxes[i]
        mx1, my1, mx2, my2 = merged_boxes[-1]

        # Calculate the area of intersection
        ix1, iy1, ix2, iy2 = max(x1, mx1), max(y1, my1), min(x2, mx2), min(y2, my2)
        inter_area = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)

        # Calculate the area of both boxes
        box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        box2_area = (mx2 - mx1 + 1) * (my2 - my1 + 1)

        # Calculate the overlap ratio
        overlap_ratio = inter_area / float(box1_area + box2_area - inter_area)

        # If there is significant overlap, merge the bounding boxes
        if overlap_ratio > overlap_threshold:
            merged_x1 = min(x1, mx1)
            merged_y1 = min(y1, my1)
            merged_x2 = max(x2, mx2)
            merged_y2 = max(y2, my2)
            merged_boxes[-1] = (merged_x1, merged_y1, merged_x2, merged_y2)
        else:
            merged_boxes.append((x1, y1, x2, y2))

    Final_Boxes = []

    for i in range(0, len(merged_boxes)):
        x1, y1, x2, y2 = merged_boxes[i]
        box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if box_area >= 50 :
            Final_Boxes.append((x1, y1, x2, y2))

    return Final_Boxes


def draw_bounding_boxes_on_rgb_image(rgb_image, bounding_boxes):
    """
    Draw bounding boxes on an RGB image.

    Parameters:
    - rgb_image (ndarray): The input RGB image.
    - bounding_boxes (list): List of bounding boxes, each defined as (x_min, y_min, x_max, y_max).

    Returns:
    - ndarray: The RGB image with bounding boxes drawn on it.
    """

    # Convert the RGB image to BGR format for use with OpenCV
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Draw each bounding box on the image
    for (x_min, y_min, x_max, y_max) in bounding_boxes:
        cv2.rectangle(bgr_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(bgr_image, f"({x_min}, {y_min})", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    2)

    # Convert the BGR image back to RGB format
    result_rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    return result_rgb_image

