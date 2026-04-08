#!/usr/bin/env python

import argparse
import numpy as np
import sys
import cv2

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from utils.processing import preprocess, postprocess
from utils.render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from utils.labels import COCOLabels
from utils.obb_postprocess import generate_labels_with_colors,visualize_detections
INPUT_NAMES = ["images"]
OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes"]#, "det_masks"]
def update(fromWidth, fromHeight, toWidth, toHeight):
    #if (fromWidth == lastWidth and fromHeight == lastHeight):
    #    return
    #lastWidth  = fromWidth
    #lastHeight = fromHeight

    scale  = min(float(toWidth) / float(fromWidth), float(toHeight) / float(fromHeight))
    offset = 0.5 * scale - 0.5

    scaleFromWidth  = -0.5 * scale * fromWidth
    scaleFromHeight = -0.5 * scale * fromHeight
    halfToWidth     = 0.5 * toWidth
    halfToHeight    = 0.5 * toHeight

    if (scale != 0.0):
        invD = 1.0 / (scale * scale)
    else:
        invD = 0.0
    A    = scale * invD

    matrix0 = (A, 0.0, -A * (scaleFromWidth + halfToWidth + offset))
    matrix1 = (0.0, A, -A * (scaleFromHeight + halfToHeight + offset))
    return matrix0, matrix1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                        choices=['dummy', 'image', 'video'],
                        default='dummy',
                        help='Run mode. \'dummy\' will send an emtpy buffer to the server to test if inference works. \'image\' will process an image. \'video\' will process a video.')
    parser.add_argument('input',
                        type=str,
                        nargs='?',
                        help='Input file to load from in image or video mode')
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
    parser.add_argument('-o',
                        '--out',
                        type=str,
                        required=False,
                        default='./yolo_output.png',
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
    parser.add_argument("--obb",
                        default=True)
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

    if FLAGS.model_info:
        # Model metadata
        try:
            metadata = triton_client.get_model_metadata(FLAGS.model)
            print(metadata)
        except InferenceServerException as ex:
            if "Request for unknown model" not in ex.message():
                print("FAILED : get_model_metadata")
                print("Got: {}".format(ex.message()))
                sys.exit(1)
            else:
                print("FAILED : get_model_metadata")
                sys.exit(1)

        # Model configuration
        try:
            config = triton_client.get_model_config(FLAGS.model)
            if not (config.config.name == FLAGS.model):
                print("FAILED: get_model_config")
                sys.exit(1)
            print(config)
        except InferenceServerException as ex:
            print("FAILED : get_model_config")
            print("Got: {}".format(ex.message()))
            sys.exit(1)

    # DUMMY MODE
    if FLAGS.mode == 'dummy':
        print("Running in 'dummy' mode")
        print("Creating emtpy buffer filled with ones...")
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, FLAGS.width, FLAGS.height], "FP32"))
        inputs[0].set_data_from_numpy(np.ones(shape=(1, 3, FLAGS.width, FLAGS.height), dtype=np.float32))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[0]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[1]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[2]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[3]))

        print("Invoking inference...")
        results = triton_client.infer(model_name=FLAGS.model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=FLAGS.client_timeout)
        if FLAGS.model_info:
            statistics = triton_client.get_inference_statistics(model_name=FLAGS.model)
            if len(statistics.model_stats) != 1:
                print("FAILED: get_inference_statistics")
                sys.exit(1)
            print(statistics)
        print("Done")

        for output in OUTPUT_NAMES:
            result = results.as_numpy(output)
            print(f"Received result buffer \"{output}\" of size {result.shape}")
            print(f"Naive buffer sum: {np.sum(result)}")

    # IMAGE MODE
    if FLAGS.mode == 'image':
        print("Running in 'image' mode")
        if not FLAGS.input:
            print("FAILED: no input image")
            sys.exit(1)
            
        labels = generate_labels_with_colors(FLAGS.labels)
        
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, FLAGS.width, FLAGS.height], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[0]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[1]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[2]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[3]))

        print("Creating buffer from image file...")
        input_image = cv2.imread(str(FLAGS.input))
        if input_image is None:
            print(f"FAILED: could not load input image {str(FLAGS.input)}")
            sys.exit(1)
        input_image_buffer = preprocess(input_image, [FLAGS.width, FLAGS.height])
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        # 参数(srcWidth, srcHeight, dstWidth, dstHeight)
        matrix0, matrix1 = update(input_image.shape[0], input_image.shape[1], FLAGS.width, FLAGS.height)
        
        inputs[0].set_data_from_numpy(input_image_buffer)

        print("Invoking inference...")
        results = triton_client.infer(model_name=FLAGS.model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=FLAGS.client_timeout)
        if FLAGS.model_info:
            statistics = triton_client.get_inference_statistics(model_name=FLAGS.model)
            if len(statistics.model_stats) != 1:
                print("FAILED: get_inference_statistics")
                sys.exit(1)
            print(statistics)
        print("Done")

        for output in OUTPUT_NAMES:
            result = results.as_numpy(output)
            print(f"Received result buffer \"{output}\" of size {result.shape}")
            print(f"Naive buffer sum: {np.sum(result)}")

        num_dets = results.as_numpy(OUTPUT_NAMES[0])
        det_boxes = results.as_numpy(OUTPUT_NAMES[1])
        det_scores = results.as_numpy(OUTPUT_NAMES[2])
        det_classes = results.as_numpy(OUTPUT_NAMES[3])
        #（100，5）(100) (100)
        #boxes = det_boxes[0, :num_dets[0][0]] #遵照tensorrt yolo的规则，我们就不除以高和宽了
        # for循环100，逐次取出
        #new_boxes = [[] for _ in range(num_dets[0][0])]
        #for i in range(num_dets[0][0]):
        #    left = boxes[i][0]
        #    top = boxes[i][1]
        #    right = boxes[i][2]
        #    bottom = boxes[i][3]
        #    import pdb;pdb.set_trace()
            # 不确定是否可以inplace
            #new_left = matrix0[0] * left + matrix0[1] * top + matrix0[2]
            #new_top = matrix1[0] * left + matrix1[1] * top + matrix1[2]
            #new_right = matrix0[0] * right + matrix0[1] * bottom + matrix0[2]
            #new_bottom = matrix1[0] * right + matrix1[1] * bottom + matrix1[2]
        #    new_boxes[i].append(new_left)
        #    new_boxes[i].append(new_top)
        #    new_boxes[i].append(new_right)
        #    new_boxes[i].append(new_bottom)
        #    new_boxes[i].append(boxes[i][4])
        if FLAGS.obb:
            boxes = det_boxes[0, :num_dets[0][0]] / np.array([FLAGS.width, FLAGS.height, FLAGS.width, FLAGS.height, 1], dtype=np.float32)
        else:
            boxes = det_boxes[0, :num_dets[0][0]] / np.array([FLAGS.width, FLAGS.height, FLAGS.width, FLAGS.height], dtype=np.float32)
        scores = det_scores[0, :num_dets[0][0]]
        classes = det_classes[0, :num_dets[0][0]].astype(np.int64)
        img_h, img_w = input_image.shape[0], input_image.shape[1]
        old_h, old_w = img_h, img_w
        offset_h, offset_w = 0, 0
        letter_box = True
        if letter_box:#below maybe width and height is reversed
            #if (img_w / FLAGS.width) >= (img_h / FLAGS.height):
            #    old_h = int(FLAGS.height * img_w / FLAGS.width)
            #    offset_h = (old_h - img_h) // 2
            #else:
            #    old_w = int(FLAGS.width * img_h / FLAGS.height)
            #    offset_w = (old_w - img_w) // 2        
            if (img_w / FLAGS.height) >= (img_h / FLAGS.width):
                old_h = int(FLAGS.width * img_w / FLAGS.height)
                offset_h = (old_h - img_h) // 2
            else:
                old_w = int(FLAGS.height * img_h / FLAGS.width)
                offset_w = (old_w - img_w) // 2
        boxes = boxes * np.array([old_w, old_h, old_w, old_h, 1], dtype=np.float32)
        if letter_box:
            if FLAGS.obb:
                boxes -= np.array([offset_w, offset_h, offset_w, offset_h, 0], dtype=np.float32)
            else:
                boxes -= np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
        #import pdb;pdb.set_trace()
        # 100个box scores classes
        #output_image = visualize_detections(input_image, new_boxes, scores, classes, labels, is_obb=True)
        output_image = visualize_detections(input_image, boxes, scores, classes, labels, is_obb=True, is_seg=False)
        # detected_objects = postprocess(num_dets, det_boxes, det_scores, det_classes, input_image.shape[1], input_image.shape[0], [FLAGS.width, FLAGS.height])
        print(f"Detected objects: {len(boxes)}")
        # for box in detected_objects:#处理一张图片里面的所有box
            # print(f"{COCOLabels(box.classID).name}: {box.confidence}")
            # input_image = render_box(input_image, box.box(), color=tuple(RAND_COLORS[box.classID % 64].tolist()))
            # size = get_text_size(input_image, f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", normalised_scaling=0.6)
            # input_image = render_filled_box(input_image, (box.x1 - 3, box.y1 - 3, box.x1 + size[0], box.y1 + size[1]), color=(220, 220, 220))
            # input_image = render_text(input_image, f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", (box.x1, box.y1), color=(30, 30, 30), normalised_scaling=0.5)

        if FLAGS.out:
            # cv2.imwrite(FLAGS.out, input_image)
            cv2.imwrite(FLAGS.out, output_image)
            print(f"Saved result to {FLAGS.out}")
        else:
            # cv2.imshow('image', input_image)
            cv2.imshow('image', output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
