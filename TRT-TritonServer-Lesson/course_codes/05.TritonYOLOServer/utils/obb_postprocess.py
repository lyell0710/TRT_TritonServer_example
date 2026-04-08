import random
from typing import List, Tuple
from pathlib import Path
import cv2
import numpy as np

# box: left, top, right, bottom, theta�convert coords of 4 dingdian
def xyxyr2xyxyxyxy(box):  # type: ignore
    """
    Convert a rotated bounding box to the coordinates of its four corners.

    Args:
        box (Box): The bounding box with left, top, right, bottom attributes,
        and a rotation angle (theta).

    Returns:
        List[Tuple[int, int]]: A list of four corner coordinates,
        each as an (x, y) tuple.
    """
    left = box[0]
    top = box[1]
    right = box[2]
    bottom = box[3]
    theta = box[4]
    # Calculate the cosine and sine of the angle
    cos_value = np.cos(theta)
    sin_value = np.sin(theta)

    # Calculate the center coordinates of the box
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5

    # Calculate the half width and half height of the box
    half_width = (right - left) * 0.5
    half_height = (bottom - top) * 0.5
    # Calculate the cosine and sine of the angle
    # cos_value = np.cos(box.theta)
    # sin_value = np.sin(box.theta)

    # # Calculate the center coordinates of the box
    # center_x = (box.left + box.right) * 0.5
    # center_y = (box.top + box.bottom) * 0.5

    # # Calculate the half width and half height of the box
    # half_width = (box.right - box.left) * 0.5
    # half_height = (box.bottom - box.top) * 0.5

    # Calculate the rotated corner vectors
    vec_x1 = half_width * cos_value
    vec_y1 = half_width * sin_value
    vec_x2 = half_height * sin_value
    vec_y2 = half_height * cos_value

    # Return the four corners of the rotated rectangle
    return [
        (int(center_x + vec_x1 - vec_x2), int(center_y + vec_y1 + vec_y2)),
        (int(center_x + vec_x1 + vec_x2), int(center_y + vec_y1 - vec_y2)),
        (int(center_x - vec_x1 + vec_x2), int(center_y - vec_y1 - vec_y2)),
        (int(center_x - vec_x1 - vec_x2), int(center_y - vec_y1 + vec_y2)),
    ]
    
def visualize_detections(
    image: np.ndarray,
    # det_result: DetectionResult,  # type: ignore
    det_boxes,
    det_scores,
    det_classes,
#    det_maskes,
    labels: List[Tuple[str, Tuple[int, int, int]]],#��~O~V�~Z~D�~S�~E��~[�~]��~Z~Dlabel.txt
    is_obb: bool,
    is_seg: bool
) -> np.ndarray:
    """
    Visualize object detections on the input image and return the result.

    Args:
        image (np.ndarray): The input image on which to visualize detections.
        det_result (DetectionResult): Object containing detection results, including the number of detections,
                                      bounding boxes, class indices, and scores.
        labels (List[Tuple[str, Tuple[int, int, int]]]): A list of label names and their corresponding RGB colors.
                                                         Each element is a tuple where the first item is the label name
                                                         and the second is the color tuple (R, G, B).
        is_obb (bool): A flag indicating whether the bounding boxes are oriented (rotated) bounding boxes (OBB).
                       If True, the bounding boxes are treated as rotated rectangles.
        is_seg (bool): A flag indicating whether the model is a segmented model.
                       If True, the bounding boxes will be extracted by segment mask.
    Returns:
        np.ndarray: The image with visualized detections, including labeled bounding boxes or rotated rectangles or masked.
    """
    # Line width, font size, and text thickness
    line_width = max(round(sum(image.shape) / 2 * 0.003), 2)
    font_thickness = max(line_width - 1, 1)
    font_scale = line_width / 3

    # Colors
    lowlight_color = (253, 168, 208)
    mediumlight_color = (251, 81, 163)
    highlight_color = (125, 40, 81)
    #for box, class_, score, mask in zip(det_boxes, det_classes, det_scores, det_maskes):
    for box, class_, score in zip(det_boxes, det_classes, det_scores):
        # import pdb;pdb.set_trace()
        color = labels[class_][1]
        label_text = f"{labels[class_][0]} {score:.2f}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        if is_obb:
            corners = xyxyr2xyxyxyxy(box)
            corners = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))

            # Draw label text
            label_top_left = (corners[0][0][0], corners[0][0][1] - label_size[1])
            label_bottom_right = (corners[0][0][0] + label_size[0], corners[0][0][1])
            cv2.rectangle(image, label_top_left, label_bottom_right, color, thickness=-1)
            cv2.putText(image, label_text, (corners[0][0][0], corners[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw polylines for rotated bounding box
            corners = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [corners], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
        elif is_seg:
            mask = None
            resized_mask = cv2.resize(mask.to_numpy(), (image_width, image_height)) > 0
            box_mask = np.zeros_like(resized_mask, dtype=bool)      
            box_mask[box[1] : box[3], box[0] : box[2]] = True
            resized_mask &= box_mask
            image[resized_mask] = image[resized_mask] * 0.5 + np.array(lowlight_color) * 0.5
            image = np.clip(image, 0, 255).astype(np.uint8)            
            # 100-112��det seg obb����������
            text_width, text_height = cv2.getTextSize(label_text, 0, fontScale=font_scale, thickness=font_thickness)[0]
            text_height += 3  # Padding
            box_top_left = (box[0], box[1])
            box_bottom_right = (box[2], box[3])
            is_text_outside = box_top_left[1] >= text_height
            if box_top_left[0] > image_width - text_width:
                box_top_left = image_width - text_width, box_top_left[1]

            box_bottom_right = box_top_left[0] + text_width, box_top_left[1] - text_height if is_text_outside else box_top_left[1] + text_height
            cv2.rectangle(image, box_top_left, box_bottom_right, highlight_color, -1, cv2.LINE_AA)

            text_position = (box_top_left[0], box_top_left[1] - 2 if is_text_outside else box_top_left[1] + text_height - 1)
            cv2.putText(image, label_text, text_position, 0, font_scale, lowlight_color, thickness=font_thickness, lineType=cv2.LINE_AA)
        else:
            left = box[0]
            top = box[1]
            right = box[2]
            bottom = box[3]
            box = list(map(int, [left, top, right, bottom]))

            # Draw label text
            label_rect = (box[0], box[1], label_size[0], label_size[1])
            # label_rect = (left, top, label_size[0], label_size[1])
            label_rect = tuple(map(int, label_rect))
            cv2.rectangle(image, label_rect[:2], (label_rect[0] + label_rect[2], label_rect[1] + label_rect[3]), color, -1)
            cv2.putText(#��~Nrender_text, box.x1,box.y1=label_rect0,label_rect1
                image, label_text, (label_rect[0], label_rect[1] + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

            # Draw bounding box
            cv2.rectangle(image, (left, top), (right, bottom), color, thickness=1, lineType=cv2.LINE_AA)

    return image

def old_visualize_detections(
    image: np.ndarray,
    # det_result: DetectionResult,  # type: ignore
    det_boxes,
    det_scores,
    det_classes,
    labels: List[Tuple[str, Tuple[int, int, int]]],#读取的输入进来的label.txt
    is_obb: bool,
) -> np.ndarray:
    """
    Visualize object detections on the input image and return the result.

    Args:
        image (np.ndarray): The input image on which to visualize detections.
        det_result (DetectionResult): Object containing detection results, including the number of detections,
                                      bounding boxes, class indices, and scores.
        labels (List[Tuple[str, Tuple[int, int, int]]]): A list of label names and their corresponding RGB colors.
                                                         Each element is a tuple where the first item is the label name
                                                         and the second is the color tuple (R, G, B).
        is_obb (bool): A flag indicating whether the bounding boxes are oriented (rotated) bounding boxes (OBB).
                       If True, the bounding boxes are treated as rotated rectangles.

    Returns:
        np.ndarray: The image with visualized detections, including labeled bounding boxes or rotated rectangles.
    """
    # Line width, font size, and text thickness
    line_width = max(round(sum(image.shape) / 2 * 0.003), 2)
    font_thickness = max(line_width - 1, 1)
    font_scale = line_width / 3

    # Colors
    lowlight_color = (253, 168, 208)
    mediumlight_color = (251, 81, 163)
    highlight_color = (125, 40, 81)
    # 这个for循环应该是在循环处理每一个box的信息 class和score
    # TODO run一下detect.py看一下det_boxes的shape，box的数据结构
    for box, class_, score in zip(det_boxes, det_classes, det_scores):
        color = labels[class_][1]
        label_text = f"{labels[class_][0]} {score:.2f}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        if is_obb:
            corners = xyxyr2xyxyxyxy(box)
            corners = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))

            # Draw label text
            label_top_left = (corners[0][0][0], corners[0][0][1] - label_size[1])
            label_bottom_right = (corners[0][0][0] + label_size[0], corners[0][0][1])
            cv2.rectangle(image, label_top_left, label_bottom_right, color, thickness=-1)
            cv2.putText(image, label_text, (corners[0][0][0], corners[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw polylines for rotated bounding box
            corners = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [corners], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
        else:
            left = box[0]
            top = box[1]
            right = box[2]
            bottom = box[3]
            box = list(map(int, [left, top, right, bottom]))

            # Draw label text
            label_rect = (box[0], box[1], label_size[0], label_size[1])
            # label_rect = (left, top, label_size[0], label_size[1])
            label_rect = tuple(map(int, label_rect))
            cv2.rectangle(image, label_rect[:2], (label_rect[0] + label_rect[2], label_rect[1] + label_rect[3]), color, -1)
            cv2.putText(#对于render_text, box.x1,box.y1=label_rect0,label_rect1
                image, label_text, (label_rect[0], label_rect[1] + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

            # Draw bounding box
            cv2.rectangle(image, (left, top), (right, bottom), color, thickness=1, lineType=cv2.LINE_AA)
        # for segmentation, TODO:need pass mask to args, so need change the func input back to det_result this struct
        # resized_mask = cv2.resize(result.masks[i].to_numpy(), (image_width, image_height)) > 0
        # box_mask = np.zeros_like(resized_mask, dtype=bool)
        # box_mask[box[1] : box[3], box[0] : box[2]] = True
        # resized_mask &= box_mask
        # image[resized_mask] = image[resized_mask] * 0.5 + np.array(lowlight_color) * 0.5
        # image = np.clip(image, 0, 255).astype(np.uint8)
        # not done
    return image

def generate_labels_with_colors(labels_file: str) -> List[Tuple[str, Tuple[int, int, int]]]:
    """
    Generate labels with random RGB colors based on the information in the labels file.

    Args:
        labels_file (str): Path to the labels file.

    Returns:
        List[Tuple[str, Tuple[int, int, int]]]: List of label-color tuples.
    """

    def generate_random_rgb() -> Tuple[int, int, int]:
        """
        Generate a random RGB color tuple.

        Returns:
            Tuple[int, int, int]: Random RGB color tuple.
        """
        return tuple(random.randint(0, 255) for _ in range(3))

    with open(labels_file) as f:
        return [(label.strip(), generate_random_rgb()) for label in f]
 
