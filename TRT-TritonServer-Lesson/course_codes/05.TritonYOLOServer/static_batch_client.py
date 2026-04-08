
import argparse
import numpy as np
import sys
import os
import cv2
from pathlib import Path
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from utils.processing import batch_preprocess, preprocess, postprocess
from utils.render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from utils.labels import COCOLabels
from utils.obb_postprocess import generate_labels_with_colors,visualize_detections
INPUT_NAMES = ["images"]
OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes"]
def get_images_in_batches(folder_path: str, batch: int, is_cuda_graph: bool):
    """
    Get a list of image files in the specified directory, grouped into batches.

    Args:
        folder_path (str): Path to the directory to search for image files.
        batch (int): The number of images in each batch.
        is_cuda_graph (bool): Flag indicating whether to discard extra images if using CUDA graph.

    Returns:
        List[List[str]]: List of image file paths grouped into batches.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [path for path in Path(folder_path).rglob('*') if path.suffix.lower() in image_extensions and path.is_file()]

    if not is_cuda_graph:
        # If not using CUDA graph, include all images, padding the last batch if necessary
        batches = [image_files[i : i + batch] for i in range(0, len(image_files), batch)]
    else:
        # If using CUDA graph, exclude extra images that don't fit into a full batch
        total_images = (len(image_files) // batch) * batch
        batches = [image_files[i : i + batch] for i in range(0, total_images, batch)]

    return batches
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
        bs = FLAGS.batch_size
        
        batchs = get_images_in_batches(FLAGS.input, bs, is_cuda_graph=True)
        for batch in batchs:
            # images = [cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB) for image_path in batch]
            images = [cv2.imread(str(image_path)) for image_path in batch]

            inputs = []
            outputs = []
            inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [bs, 3, FLAGS.width, FLAGS.height], "FP32"))
            outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[0]))
            outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[1]))
            outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[2]))
            outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[3]))

            print("Creating batch buffer from {len(images)} image file...")
            input_image_buffer = batch_preprocess(images, [FLAGS.width, FLAGS.height])
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
            #import pdb;pdb.set_trace()
            if FLAGS.obb: #[bs,100,5],不太确定:num_dets[0][0]在这里是否正确，期待的是拿到[bs,100,5]
                boxes = det_boxes[:bs, :num_dets[0][0]] / np.array([FLAGS.width, FLAGS.height, FLAGS.width, FLAGS.height, 1], dtype=np.float32)
            else:
                boxes = det_boxes[:bs, :num_dets[0][0]] / np.array([FLAGS.width, FLAGS.height, FLAGS.width, FLAGS.height], dtype=np.float32)
            scores = det_scores[:bs, :num_dets[0][0]]
            classes = det_classes[:bs, :num_dets[0][0]].astype(np.int64)
            for i, input_image in enumerate(images):
                img_h, img_w = input_image.shape[0], input_image.shape[1]
                old_h, old_w = img_h, img_w
                offset_h, offset_w = 0, 0
                letter_box = True
                if letter_box:#below maybe width and height is reversed
                    if (img_w / FLAGS.height) >= (img_h / FLAGS.width):
                        old_h = int(FLAGS.width * img_w / FLAGS.height)
                        offset_h = (old_h - img_h) // 2
                    else:
                        old_w = int(FLAGS.height * img_h / FLAGS.width)
                        offset_w = (old_w - img_w) // 2
                # boxes的shape期待是[100,5]
                if FLAGS.obb:
                    boxes[i] = boxes[i] * np.array([old_w, old_h, old_w, old_h, 1], dtype=np.float32)
                else:
                    boxes[i] = boxes[i] * np.array([old_w, old_h, old_w, old_h], dtype=np.float32)
                if letter_box:
                    if FLAGS.obb:
                        boxes[i] -= np.array([offset_w, offset_h, offset_w, offset_h, 0], dtype=np.float32)
                    else:
                        boxes[i] -= np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
                # 100 box scores classes
                #import pdb;pdb.set_trace()
                print("visualizing ", batch[i])
                output_image = visualize_detections(input_image, boxes[i], scores[i], classes[i], labels, is_obb=True, is_seg=False)
                # output_image = batch_visualize_detections(images, boxes, scores, classes, labels, is_obb=True)

                print(f"Detected objects: {len(boxes[i])}")
                if FLAGS.out_dir:
                    os.makedirs(FLAGS.out_dir, exist_ok=True)
                    cv2.imwrite(str(FLAGS.out_dir + "/" + str(batch[i]).split("/")[-1]), output_image)
                    #cv2.imwrite(str(FLAGS.out_dir / batch[i]), output_image)
                    print(f"Saved result to {FLAGS.out_dir}")
                else:
                    cv2.imshow('image', output_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

