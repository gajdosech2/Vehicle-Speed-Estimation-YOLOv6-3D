import argparse
import os
import time
import random
from queue import Queue, Empty
from threading import Event, Thread

import cv2
import numpy as np

from transform_3D_utils.radar import Radar
from yolov6.core.inferer import Inferer
from yolov6.utils.events import LOGGER
from yolov6.trt_inferer import TrtInferer
from transform_3D_utils.utils import get_calibration_params, compute_camera_calibration, \
    get_transform_matrix_with_criterion


TIMEOUT = 200

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Yolov6 3d speed measurement', add_help=add_help)
    parser.add_argument('--tensorrt', action='store_true', help='If model for inference is TensorRT optimized.')
    parser.add_argument('--model_path', type=str, default='checkpoints/nano/yolov6_nano_3d_transform.pt', help='model path(s) for inference.')
    parser.add_argument('--yolo-img-size', nargs='+', type=int, default=[270, 480],
                        help='the image-size(h,w) in inference size.')
    parser.add_argument('--img-size', nargs='+', type=int, default=[960, 540],
                        help='The image size (h,w) for inference.')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold for inference.')
    parser.add_argument('--half', action='store_true',
                        help='whether to use FP16 half-precision inference.')
    parser.add_argument('--show-video', action='store_true', help='Show video of inference with 3D bouding boxes.')
    parser.add_argument('--video-fps', type=int, default=50, help='Video FPS')
    parser.add_argument('--test-name', type=str, default='yolov6_3d_nano', help='Test name')
    parser.add_argument('--result-dir', type=str, default='', help='Result directory')
    parser.add_argument('--processing-batch', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--root_dir_video_path', type=str, default='',
                        help='Root directory of videos. Where are sessions folders located')
    parser.add_argument('--root_dir_results_path', type=str, default='',
                        help='Root directory of results. For each test the directory will be created')
    args = parser.parse_args()
    LOGGER.info(args)
    return args


def load_test_videos(root_dir_video_path: str, root_dir_results_path: str):
    vid_list = []
    calib_list = []
    store_results_list = []
    road_mask_list = []
    for i in range(6, 7):
        dir_list = ['session{}_center'.format(i), 'session{}_left'.format(i), ]
        vid_list.extend([os.path.join(root_dir_video_path, d, 'video.avi') for d in dir_list])
        road_mask_list.extend([os.path.join(root_dir_video_path, d, 'video_mask.png') for d in dir_list])
        calib_list.extend(
            [os.path.join(root_dir_results_path, d, 'system_SochorCVIU_Edgelets_BBScale_Reg.json') for d in dir_list])
        store_results_list.extend([os.path.join(root_dir_results_path, d) for d in dir_list])
    return vid_list, calib_list, store_results_list, road_mask_list


def batch_process_video(inferer: Inferer,
                        camera_calibration_file: str,
                        video_path: str,
                        road_mask_path: str,
                        img_size: tuple,
                        yolo_img_size: tuple,
                        result_dir: str,
                        test_name: str,
                        show_video: bool = True,
                        batch_size_processing: int = 32,
                        video_fps: int = 50,
                        iou_threshold: float = 0.65,
                        conf_threshold: float = 0.65, ):
    avg_fps = []
    im_w, im_h = img_size
    camera_calibration = get_calibration_params(camera_calibration_file)
    vp1, vp2, vp3, pp, road_plane, focal = compute_camera_calibration(
        [camera_calibration['vp1'], camera_calibration['vp2']], camera_calibration['pp']
    )
    vp1 = vp1[:-1] / vp1[-1]
    vp2 = vp2[:-1] / vp2[-1]
    vp3 = vp3[:-1] / vp3[-1]

    cap = cv2.VideoCapture(video_path)
    road_mask = cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE)

    M, IM = get_transform_matrix_with_criterion(vp3, vp2, road_mask, im_w, im_h, constraint=0.8,
                                                vp_top=None)

    vp0_t = np.array([vp1], dtype="float32")
    vp0_t = np.array([vp0_t])
    vp0_t = cv2.perspectiveTransform(vp0_t, M)
    vp0_t = vp0_t[0][0]

    warp_perspective_lambda = lambda image, transform_matrix, img_size: cv2.warpPerspective(
        image,
        transform_matrix,
        (img_size[0], img_size[1]),
        borderMode=cv2.BORDER_CONSTANT,
    )
    radar = Radar(transform_matrix=M,
                  inv_transform_matrix=IM,
                  vanishing_points=[vp1, vp2, vp3],
                  vp0_t=vp0_t,
                  image_size=img_size,
                  projector=None,
                  video_fps=video_fps,
                  result_path=result_dir,
                  result_name=test_name,
                  camera_calib_structure=camera_calibration)

    if isinstance(inferer, TrtInferer):
        lambda_inferer = lambda images: inferer.infer(images)
    else:
        lambda_inferer = lambda images: inferer.simple_inference(images, conf_threshold, iou_threshold)

    q_frames = Queue(batch_size_processing)
    q_images = Queue(batch_size_processing)
    q_predict = Queue(batch_size_processing)
    e_stop = Event()

    def read_frames():
        frames_count = 0
        while cap.isOpened() and not e_stop.is_set():
            images = []
            frames = []
            for _ in range(batch_size_processing):
                ret, frame = cap.read()
                frames_count += 1
                if not ret or frame is None or frames_count > 30000:
                    cap.release()
                    if len(images) > 0:
                        q_images.put(images)
                        q_frames.put(frames)
                    q_images.put(None)
                    q_frames.put(None)
                    break
                frames.append(frame)
                image = cv2.bitwise_and(frame, frame, mask=road_mask)
                t_image = warp_perspective_lambda(image, M, (im_w, im_h))
                images.append(t_image)
            q_images.put(images)
            q_frames.put(frames)

    def predict():
        while not e_stop.is_set():
            try:
                images = q_images.get(timeout=TIMEOUT)
                if images is None:
                    e_stop.set()
                    break
            except Empty:
                e_stop.set()
                break
            gpu_time = time.time()
            images = np.stack(images, axis=0)
            bbox_2d, fub = lambda_inferer(images)
            q_predict.put((bbox_2d, fub))
            gpu_finish_time = (time.time() - gpu_time)
            avg_fps.append(batch_size_processing / gpu_finish_time)
            if random.randrange(0, 50) == 1:
                print("GPU FPS: {}".format(batch_size_processing / gpu_finish_time))

    def process():
        while not e_stop.is_set():
            try:
                frames = q_frames.get(timeout=TIMEOUT)
                bbox_2d, fub = q_predict.get(timeout=TIMEOUT)
                if frames is None:
                    e_stop.set()
                    break
            except Empty:
                e_stop.set()
                break
            for i, (frame, box, f) in enumerate(zip(frames, bbox_2d, fub)):
                image_b = radar.process_frame(box, f, frame)
                #cv2.imwrite("/home/g/gajdosech2/YOLOv6_transform_3d/debug_pt.jpg", image_b)
                if show_video:
                    cv2.imshow('frame', image_b)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        e_stop.set()

    reader = Thread(target=read_frames)
    predictor = Thread(target=predict)
    processor = Thread(target=process)

    reader.start()
    predictor.start()
    processor.start()

    reader.join()
    predictor.join()
    processor.join()
    mean_fps = int(np.mean(avg_fps))
    with open(os.path.join(result_dir, 'avg_fps_' + test_name + '.txt'), 'a') as f:
        f.write("Average GPU time:" + str(mean_fps) + ", for " + video_path + "\n")
    print("Average GPU time:", mean_fps, ", for ", video_path)


if __name__ == "__main__":
    args = get_args_parser()
    root_dir_video_path = args.root_dir_video_path
    root_dir_results_path = args.root_dir_results_path
    vid_list, calib_list, store_results_list, road_mask_list = load_test_videos(root_dir_video_path,
                                                                                root_dir_results_path)
    if args.model_path.endswith(".trt"):
        inferer = TrtInferer(trt_model=args.model_path, image_size=args.yolo_img_size, half=args.half)
    else:
        inferer = Inferer(model=args.model_path, yolo_img_size=args.yolo_img_size, half=args.half)

    for vid_path, calib_path, store_results_path, mask_path in zip(vid_list, calib_list, store_results_list,
                                                                   road_mask_list):
        start_processing = time.time()
        print("Processing: {}".format(vid_path))
        batch_process_video(inferer,
                            calib_path,
                            vid_path,
                            mask_path,
                            args.img_size,
                            args.yolo_img_size,
                            store_results_path,
                            args.test_name,
                            args.show_video,
                            args.processing_batch,
                            args.video_fps,
                            args.iou_thres)
        print("Finished. Processing time: {}".format(time.time() - start_processing))
