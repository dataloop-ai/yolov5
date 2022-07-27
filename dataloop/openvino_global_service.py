import logging
import dtlpy as dl
import numpy as np
import cv2
import torch
import sys

sys.path.append('/app')
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

ROOT = './dataloop/tmp/weights'
LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
          'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
          'hair drier', 'toothbrush']
logger = logging.getLogger('YOLOV5')


class Yolov5Openvino(dl.BaseServiceRunner):
    def __init__(self):
        weights = './dataloop_tmp/weights/yolov5s_openvino_model_640_640/yolov5s.xml'
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = 'cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img = False  # show results
        self.save_crop = False  # save cropped prediction boxes
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.line_thickness = 3  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference
        device = select_device(self.device)
        model = DetectMultiBackend(weights, device=device, dnn=self.dnn)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Half
        self.half &= (pt or jit or onnx or engine) and device.type != 'cpu'
        if pt or jit:
            model.model.half() if self.half else model.model.float()
        # Run inference
        model.warmup(imgsz=(1, 3, *imgsz), half=self.half)  # warmup
        self.model = model
        logger.info('Model loaded and ready to go!')

    def preprocess(self, x):
        # Padded resize
        img = letterbox(x,
                        self.imgsz,
                        stride=self.model.stride,
                        auto=self.model.pt
                        )[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img).astype(float)
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def run(self, item: dl.Item, progress=None, config=None):

        dt = [0.0, 0.0, 0.0]

        t1 = time_sync()
        batch = np.asarray([cv2.imread(item.download(overwrite=True))])
        preprocessed_batch = torch.from_numpy(np.asarray([self.preprocess(img) for img in batch])).to(self.device)
        preprocessed_batch = preprocessed_batch.half() if self.half else preprocessed_batch.float()  # uint8 to fp16/32
        logger.info('[preprocess]: model batch size{}'.format(preprocessed_batch.shape))
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = False
        pred = self.model(preprocessed_batch, augment=self.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred,
                                   self.conf_thres,
                                   self.iou_thres,
                                   self.classes,
                                   self.agnostic_nms,
                                   max_det=self.max_det)
        dt[2] += time_sync() - t3
        batch_annotations = list()
        for i_img, det in enumerate(pred):  # per image
            img_shape = batch[i_img].shape
            p_img_shape = preprocessed_batch[i_img].shape[1:]
            det[:, :4] = scale_coords(p_img_shape, det[:, :4], img_shape).round()
            image_annotations = dl.AnnotationCollection()
            for *xyxy, conf, cls in reversed(det):
                image_annotations.add(annotation_definition=dl.Box(left=xyxy[0],
                                                                   top=xyxy[1],
                                                                   right=xyxy[2],
                                                                   bottom=xyxy[3],
                                                                   label=LABELS[int(cls)]
                                                                   # when loading snapshot, json treats keys as str
                                                                   ),
                                      model_info={'name': 'yolov5-openvino',
                                                  'confidence': conf})
            batch_annotations.append(image_annotations)

        # Process predictions

        # Print results
        seen = len(batch_annotations)
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        logger.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {batch.shape}' % t)
        return batch_annotations[0]


def deploy_package():
    import os
    import dtlpy as dl
    func = [dl.PackageFunction(name='run',
                               inputs=[dl.FunctionIO(type="Item", name="item")],
                               description='Inference on pre-trained YOLO V5'
                               )
            ]
    modules = [dl.PackageModule(entry_point='dataloop/openvino_global_service.py',
                                functions=func)]

    slots = [
        dl.PackageSlot(
            module_name="default_module",
            function_name="run",
            display_name="Item Auto Annotation YOLO V5",
            display_icon='fas fa-magic',
            post_action=dl.SlotPostAction(
                type=dl.SlotPostActionType.DRAW_ANNOTATION),
            display_scopes=[
                dl.SlotDisplayScope(
                    resource=dl.SlotDisplayScopeResource.ITEM,
                    filters=dl.Filters(
                        resource=dl.FiltersResource.ITEM)
                )
            ],

        )
    ]

    package_name = 'model-annotation-yolov5'
    project_name = 'DataloopTasks'

    project = dl.projects.get(project_name=project_name)
    ################
    # push package #
    ################
    package = project.packages.push(package_name=package_name,
                                    modules=modules,
                                    slots=slots,
                                    is_global=True,
                                    src_path=os.getcwd(),
                                    ignore_sanity_check=True)

    # package = project.packages.get(package_name=package_name)
    #####################
    # create service #
    #####################
    service = package.services.deploy(service_name=package_name,
                                      runtime={'gpu': False,
                                               'numReplicas': 1,
                                               'podType': 'regular-s',
                                               'concurrency': 20,
                                               'runnerImage': 'dataloop_runner-cpu/yolov5-openvino:2'},
                                      is_global=True,
                                      jwt_forward=True,
                                      bot='pipelines-reg@dataloop.ai')

    ##########
    # Update #
    ##########
    # # update the service to the new package code
    service = package.services.get(service_name=package.name.lower())
    service.package_revision = package.version
    service.update(True)


def local_test():
    r = Yolov5Openvino()
    dl.setenv('prod')
    item = dl.items.get(item_id='62dfe419bbacee457fedd7c2')
    annotations = r.run(item=item)
    item.annotations.upload(annotations)
