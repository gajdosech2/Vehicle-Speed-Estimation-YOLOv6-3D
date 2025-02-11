import argparse
import cv2
import numpy as np
import os
import copy
from yolov6.core.inferer import Inferer

import sys
sys.path.append("/home/photoneo/YOLOv6_transform_3d/")

from transform_3D_utils.utils import get_calibration_params, compute_camera_calibration, \
    get_transform_matrix_with_criterion


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Calibration Data', add_help=add_help)
    parser.add_argument('--root_dir_video_path', type=str, default='/home/photoneo/2016-ITS-BrnoCompSpeed/dataset/',
                        help='Root directory of videos. Where are sessions folders located')
    parser.add_argument('--root_dir_results_path', type=str, default='/home/photoneo/2016-ITS-BrnoCompSpeed/results_viktor/',
                        help='Root directory of results. For each test the directory will be created')
    parser.add_argument('--model_path', type=str, default='/home/photoneo/YOLOv6_transform_3d/bcs_trained_models/qa_small/yolov6_qa_small_3d_transform.pt', help='model path(s) for inference.')
    parser.add_argument('--img-size', nargs='+', type=int, default=[960, 540],
                        help='The image size (h,w) for inference.')
    parser.add_argument('--yolo-img-size', nargs='+', type=int, default=[480, 480],
                        help='the image-size(h,w) in inference size.')
    args = parser.parse_args()
    return args


def load_train_videos(root_dir_video_path: str, root_dir_results_path: str):
    vid_list = []
    calib_list = []
    store_results_list = []
    road_mask_list = []
    for i in range(0, 3):
        dir_list = ['session{}_center'.format(i), 'session{}_left'.format(i), 'session{}_right'.format(i)]
        vid_list.extend([os.path.join(root_dir_video_path, d, 'video.avi') for d in dir_list])
        road_mask_list.extend([os.path.join(root_dir_video_path, d, 'video_mask.png') for d in dir_list])
        calib_list.extend(
            [os.path.join(root_dir_results_path, d, 'system_SochorCVIU_Edgelets_ManualScale.json') for d in dir_list])
        store_results_list.extend([os.path.join(root_dir_results_path, d) for d in dir_list])
    return vid_list, calib_list, store_results_list, road_mask_list


if __name__ == "__main__":
    args = get_args_parser()
    vid_list, calib_list, store_results_list, road_mask_list = load_train_videos(args.root_dir_video_path,
                                                                                args.root_dir_results_path)

    calib_image_count = 0
    inferer = Inferer(model=args.model_path, yolo_img_size=args.yolo_img_size, half=True)
    
    for vid_path, calib_path, store_results_path, mask_path in zip(vid_list, calib_list, store_results_list,
                                                                   road_mask_list):

        im_w, im_h = args.img_size
        camera_calibration = get_calibration_params(calib_path)
        vp1, vp2, vp3, pp, road_plane, focal = compute_camera_calibration(
            [camera_calibration['vp1'], camera_calibration['vp2']], camera_calibration['pp']
        )
        vp1 = vp1[:-1] / vp1[-1]
        vp2 = vp2[:-1] / vp2[-1]
        vp3 = vp3[:-1] / vp3[-1]

        cap = cv2.VideoCapture(vid_path)
        road_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

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

        while cap.isOpened() and calib_image_count < 1000000:
            calib_image_count += 1

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                break

            if calib_image_count % 50 != 0:
                continue

            image = cv2.bitwise_and(frame, frame, mask=road_mask)
            t_image = warp_perspective_lambda(image, M, (im_w, im_h))

            images = np.stack([copy.deepcopy(t_image)], axis=0)
            bbox_2d, fub = inferer.simple_inference(images, 0.65, 0.65)

            latterbox, _ = Inferer.process_image(t_image, args.yolo_img_size, 32, True)
            latterbox = latterbox.detach().cpu().numpy() * 255
            latterbox = latterbox.transpose((1, 2, 0))

            if len(bbox_2d[0]) > 0:
                print(calib_image_count)
                cv2.imwrite("/home/photoneo/YOLOv6_transform_3d/calib_data/boxed" + str(calib_image_count) + ".jpg", latterbox)
            else:
                print("no bbox!")