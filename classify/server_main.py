import argparse
import os
from pathlib import Path
import datetime
import urllib.request
import shutil
import logging
import tempfile
from shapely.geometry import Polygon
from shapely.geometry import Point

import torch

from .deep_sort.tracker import Tracker
from .deep_sort import nn_matching
from .my_utils.encoder import create_box_encoder
from .models.experimental import attempt_load
from .my_utils.my_dataset import LoadImages
from .my_utils import utils
from .deep_sort import detection


import ctypes
exlibpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/lib/'
# exlibpath = '/home/site/wwwroot/lib/'
ctypes.CDLL(exlibpath + 'libglib-2.0.so.0')
ctypes.CDLL(exlibpath + 'libgthread-2.0.so.0')
import cv2

imgsz =640
def runner(video_url, queue_polygon):
    tempFilePath = tempfile.gettempdir()
    file_name = os.path.join(tempFilePath, "target_video.mp4")
    # Download the file from `url` and save it locally under `file_name`:
    with urllib.request.urlopen(video_url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    logging.info(os.getcwd())
    
    return run(weights='classify/yolov5s_custom.pt', source=file_name, output_dir=tempFilePath, queue_polygon=queue_polygon)
def run(
    weights='yolov5s.pt',  # model.pt path(s)
    source='frames',  # file/dir/URL/glob, 0 for webcam
    output_dir='out', 
    device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    conf_thres=0.5,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    queue_polygon=[181,568,936,350,1071,655,252,928],   # x y x y x y x y x y
    displacement_thres=10,
    debug_frames=0, # debug mode
    half=False,  # use FP16 half-precision inference
    save_img=False,
    save_video=False,
):
    vertices = []
    for x, y in zip(*[iter(queue_polygon)]*2):   # loop 2 coords at a time
        vertices.append((x,y))
    queue_polygon = Polygon(vertices)
    device = utils.select_device(device)
    use_gpu = device == torch.device('cuda:0')
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    model.conf = conf_thres
    model.iou = iou_thres
    if half:
        model.half()
    
    encoder = create_box_encoder('classify/mars-small128.pb', batch_size=32)
    max_cosine_distance = 0.2
    nn_budget = None
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    dataset = LoadImages(source, img_size=640, stride=stride)

    memory = {}
    ppl_count = -1
    avg_queue_length = 0
    dir_path = Path(output_dir)
    file_path = Path('output.txt')
    dir_path.mkdir(exist_ok=True)
    p = Path(output_dir) / file_path
    if use_gpu:
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    with p.open('w') as f:
        logging.info("opening file")
        if save_video:
            for _, img, im0s, _, frame_idx in dataset:
                height, width, _ = im0s.shape
                break
            size = (width, height)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_writer = cv2.VideoWriter(str(dir_path / Path("out.mp4")), fourcc, 60, size)
        logging.info("Frames = "+ str(dataset.frames))
        for _, img, im0s, _, frame_idx in dataset:
            if (frame_idx % 15 == 0):
                logging.info(frame_idx)
            if debug_frames > 0 and frame_idx > debug_frames:
                break
            
            indexIDs = []
            boxes = []
            previous = memory.copy() # last frame we have these ppl boxes
            memory = {}

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            bgr_image = im0s

            
            results = model(img)[0]
            results = utils.non_max_suppression(results)
            if(results[0].shape[0]==0):
                continue

            det = results[0]
            det[:, :4] = utils.scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            person_ind = [i for i, cls in enumerate(det[:, -1]) if int(cls) == 0]   
            xyxy = det[person_ind, :-2]  # find person only
            # xyxy = det[:,:-2]
            xywh_boxes = utils.xyxy2xywh(xyxy)
            tlwh_boxes = utils.xywh2tlwh(xywh_boxes)
            confidence = det[:, -2]
            if use_gpu:
                tlwh_boxes = tlwh_boxes.cpu()
            features = encoder(bgr_image, tlwh_boxes)
            
            detections = [detection.Detection(bbox, confidence, 'person', feature) for bbox, confidence, feature in zip(tlwh_boxes, confidence, features)]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)


            ppl_count = 0   
            queue_list = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                indexIDs.append(track.track_id) # # this frame we have these ppl idxs
                memory[track.track_id] = [bbox[0], bbox[1], bbox[2], bbox[3]]  # this frame we have these ppl boxes
                ppl_count += 1
                if save_img or save_video:
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)
                    cv2.rectangle(bgr_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                    cv2.putText(bgr_image, "ID: " + str(track.track_id), (int(center_x), int(center_y)), 0,
                                        1e-3 * bgr_image.shape[0], (0, 255, 0), 1)


            if ppl_count > 0:
                for i, box in enumerate(boxes):
                    if indexIDs[i] in previous:
                        track_id = indexIDs[i]
                        center_x = int((box[0] + box[2]) / 2)
                        center_y = int((box[1] + box[3]) / 2)
                        p0 = (center_x, center_y)
                        previous_box = previous[track_id]
                        center_x2 = int((previous_box[0] + previous_box[2]) / 2)
                        center_y2 = int((previous_box[1] + previous_box[3]) / 2)
                        p1 = (center_x2, center_y2)
                        displacement = abs(p0[0] - p1[0]) + abs(p0[1] - p1[1])
                        if displacement < displacement_thres and queue_polygon.intersects(Point(center_x, center_y)): 
                            queue_list.append(track_id)
                avg_queue_length = (avg_queue_length + len(queue_list)) / 2
               
    return {"avg_queue_len": round(avg_queue_length)}
                

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--output-dir', type=str, default='out', help='dir for ouput files')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--line', nargs='+', type=int, default=[0, 300, 1000, 200], help='boundary crossing line')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--debug-frames', type=int, default=0, help='debug mode, run till frame number x')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference, supported on CUDA only')
    parser.add_argument('--save-img', action='store_true', help='save detection output as image')
    parser.add_argument('--save-video', action='store_true', help='save detection output as an video')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)