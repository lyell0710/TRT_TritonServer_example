import argparse
import numpy as np
import sys
import os
import cv2
import time
import queue
from pathlib import Path
from functools import partial
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from utils.processing import batch_preprocess, preprocess, postprocess
from utils.render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from utils.labels import COCOLabels
from utils.obb_postprocess import generate_labels_with_colors,visualize_detections
INPUT_NAMES = ["images"]
OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes", "det_masks"]
result_queue = queue.Queue()
def callback(rid,result,error):
    print(f"request {rid} is handling")
    req_id = result.get_response().id
    if error:
        result_queue.put(error)
    else:
        result_queue.put(result)

def get_images_from_path(folder_path: str):
    """
    Get a list of image files in the specified directory.

    Args:
        folder_path (str): Path to the directory to search for image files.

    Returns:
        List[List[str]]: List of image file paths.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [path for path in Path(folder_path).rglob('*') if path.suffix.lower() in image_extensions and path.is_file()]
    
    return image_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                        choices=['dummy', 'image', 'video'],
                        default='dummy',
                        help='Run mode. \'dummy\' will send an emtpy buffer to the server to test if inference works. \'image\' will process an image. \'video\' will process a video.')
    parser.add_argument('input',
                        type=str,
                        nargs='?',
                        help='input images folder')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=False,
                        default='yolov7',
                        help='Inference model name, default yolov7')
    parser.add_argument('--width',
                        type=int,
                        required=False,
                        default=640,
                        help='Inference model input width, default 640')
    parser.add_argument('--height',
                        type=int,
                        required=False,
                        default=640,
                        help='Inference model input height, default 640')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL, default localhost:8001')
    # parser.add_argument('-o',
    #                     '--out',
    #                     type=str,
    #                     required=False,
    #                     default='./yolo_output.png',
    #                     help='Write output into file instead of displaying it')
    parser.add_argument('-o',
                        '--out_dir',
                        type=str,
                        required=False,
                        default='./yolo11_output',
                        help='Write output into file instead of displaying it')
    parser.add_argument('-f',
                        '--fps',
                        type=float,
                        required=False,
                        default=24.0,
                        help='Video output fps, default 24.0 FPS')
    parser.add_argument('-i',
                        '--model-info',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Print model status, configuration and statistics')
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose client output')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds, default no timeout')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable SSL encrypted channel to the server')
    parser.add_argument('-r',
                        '--root-certificates',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded root certificates, default none')
    parser.add_argument('-p',
                        '--private-key',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded private key, default is none')
    parser.add_argument('-x',
                        '--certificate-chain',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded certicate chain default is none')
    parser.add_argument("-l",
                        "--labels",
                        default="../labels_obb.txt",
                        help="File to use for reading the class labels from, default: ./labels.txt")
    parser.add_argument("-bs",
                        "--batch_size",
                        default=2,
                        help="File to use for reading the class labels from, default: ./labels.txt")
    parser.add_argument("--obb",
                        default=True)
    parser.add_argument("--seg",
                        default=False)
    FLAGS = parser.parse_args()

    # Create server context
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose,
            ssl=FLAGS.ssl,
            root_certificates=FLAGS.root_certificates,
            private_key=FLAGS.private_key,
            certificate_chain=FLAGS.certificate_chain)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()
    # Health check
    if not triton_client.is_server_live():
        print("FAILED : is_server_live")
        sys.exit(1)

    if not triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        sys.exit(1)

    if not triton_client.is_model_ready(FLAGS.model):
        print("FAILED : is_model_ready")
        sys.exit(1)
    # IMAGE MODE
    if FLAGS.mode == 'image':
        print("Running in 'image' mode")
        if not FLAGS.input:
            print("FAILED: no input image")
            sys.exit(1)
        #print(1)
        labels = generate_labels_with_colors(FLAGS.labels)
        # bs = FLAGS.batch_size
        #print(2)
        image_paths = get_images_from_path(FLAGS.input)
        #print(3)
        #import pdb;pdb.set_trace()
        for i, image_path in enumerate(image_paths):
            input_image = cv2.imread(str(image_path))
            inputs = []
            outputs = []
            inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, FLAGS.width, FLAGS.height], "FP32"))
            outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[0]))
            outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[1]))
            outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[2]))
            outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[3]))  
            if FLAGS.seg:
                outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[4]))
            print("Creating buffer from image file...")
            if input_image is None:
                print(f"FAILED: could not load input image {str(FLAGS.input)}")
                sys.exit(1)
            input_image_buffer = preprocess(input_image, [FLAGS.width, FLAGS.height])
            input_image_buffer = np.expand_dims(input_image_buffer, axis=0)      
            inputs[0].set_data_from_numpy(input_image_buffer)
            print("Invoking inference...")
            # results = triton_client.infer(model_name=FLAGS.model,
            #                             inputs=inputs,
            #                             outputs=outputs,
            #                             client_timeout=FLAGS.client_timeout)    
# def test_performance(client,
#                      images,
#                      delays,
#                      FLAGS):
            async_requests = []
            time.sleep(0.1)
            # 异步推理
            #async_results = 
            triton_client.async_infer(FLAGS.model, # 模型名字，ep yolo11
                                        inputs=inputs,
                                        outputs=outputs,
                                        callback=partial(callback,i),
                                        request_id=str(i))
            #import pdb;pdb.set_trace()
            #async_requests.append(async_results)
            if FLAGS.model_info:
                statistics = triton_client.get_inference_statistics(model_name=FLAGS.model_name)
                if len(statistics.model_stats) != 1:
                    print("FAILED: get_inference_statistics")
                    sys.exit(1)
                print(statistics)
            print("Done")
            #for output in OUTPUT_NAMES:
            #    result = results.as_numpy(output)
            #    print(f"Received result buffer \"{output}\" of size {result.shape}")
            #    print(f"Naive buffer sum: {np.sum(result)}")
            try:
                result = result_queue.get(timeout=60)  # 设置超时时间为 60 秒
                if isinstance(result, Exception):
                    raise result
                #results = async_results.get_result()
                num_dets = result.as_numpy(OUTPUT_NAMES[0])
                det_boxes = result.as_numpy(OUTPUT_NAMES[1])
                det_scores = result.as_numpy(OUTPUT_NAMES[2])
                det_classes = result.as_numpy(OUTPUT_NAMES[3])
                if FLAGS.seg:
                    det_masks = result.as_numpy(OUTPUT_NAMES[4])
            #except queue.Empty:
                #print("Timeout for waiting result")
            #except Exception as e:
                #print(f"error occurred: {e}")

                print(f"det_boxes.shape = {det_boxes.shape}")
                print(f"this epoch runs {det_boxes.shape[0]} requests")
                #import pdb;pdb.set_trace()
                if FLAGS.obb:
                    boxes = det_boxes[:det_boxes.shape[0], :num_dets[0][0]] / np.array([FLAGS.width, FLAGS.height, FLAGS.width, FLAGS.height, 1], dtype=np.float32)
                else:
                    boxes = det_boxes[:det_boxes.shape[0], :num_dets[0][0]] / np.array([FLAGS.width, FLAGS.height, FLAGS.width, FLAGS.height], dtype=np.float32)
                scores = det_scores[:det_boxes.shape[0], :num_dets[0][0]]
                classes = det_classes[:det_boxes.shape[0], :num_dets[0][0]].astype(np.int64)
                img_h, img_w = input_image.shape[0], input_image.shape[1]
                old_h, old_w = img_h, img_w # 当前输入图像的尺寸
                offset_h, offset_w = 0, 0
                letter_box = True
                if letter_box:# 把box放到原图像的位置
                    if (img_w / FLAGS.height) >= (img_h / FLAGS.width): # FLAGS为yolo要求的图片尺寸
                        old_h = int(FLAGS.width * img_w / FLAGS.height) # 原图在等比例缩放后的尺寸
                        offset_h = (old_h - img_h) // 2 # box需要的平移量，因为是等比例缩放，所以图片部分其实大小一样，只是letterbox后有画布使得对齐640x640，所以只需要移动画布的1/2（高或宽，注意是高或宽）
                    else:
                        old_w = int(FLAGS.height * img_h / FLAGS.width)
                        offset_w = (old_w - img_w) // 2
                #boxes = boxes * np.array([old_w, old_h, old_w, old_h, 1], dtype=np.float32)
                if letter_box:
                    if FLAGS.obb:
                        boxes = boxes * np.array([old_w, old_h, old_w, old_h, 1], dtype=np.float32)
                        boxes -= np.array([offset_w, offset_h, offset_w, offset_h, 0], dtype=np.float32)
                    else:
                        boxes = boxes * np.array([old_w, old_h, old_w, old_h], dtype=np.float32)
                        boxes -= np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
            # 可视化
                # import pdb;pdb.set_trace()
                
                output_image = visualize_detections(input_image, boxes[0], scores[0], classes[0], labels, is_obb=FLAGS.obb, is_seg=FLAGS.seg)
                print(f"Detected objects: {len(boxes)}")

                if FLAGS.out_dir:
                    os.makedirs(FLAGS.out_dir, exist_ok=True)
                    cv2.imwrite(str(FLAGS.out_dir + "/" + str(i)+".png"), output_image)
                    #cv2.imwrite(FLAGS.out_dir, output_image)
                    print(f"Saved result to {FLAGS.out_dir}")
                else:
                    cv2.imshow('image', output_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            except queue.Empty:
                print("Timeout for waiting result")
            except Exception as e:
                print(f"error occurred: {e}")
