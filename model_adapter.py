import dtlpy as dl
from dtlpy import ml
import os
import shutil
import numpy as np
import logging
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import tqdm
import json
import time
from pathlib import Path
from string import Template
import yaml
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock
import traceback
# Package specific imports
from utils.general import increment_path, non_max_suppression
from utils.callbacks import Callbacks
from utils.augmentations import letterbox


class ModelAdapter(dl.BaseModelAdapter):
    """
    Yolo5 Model adapter - based on ultralytics pytorch implementation.
    The class bind Dataloop model and snapshot entities with model code implementation


    # NOTE: Starting dtlpy version 1.35 we use a different BaseModelAdapter
            This is the updated version of the adapter for dtlpy 1.35
    """

    configuration = {
        'input_shape': (320, 640),  #  H W (480, 640),
        'model_fname': 'yolov5l.pt',
        # Detection configs
        'agnostic_nms': False,  # help='class-agnostic NMS')
        'iou_thres': 0.5,  # help='IOU threshold for NMS')
        # yaml files
        'hyp_yaml_fname': 'data/hyps/hyp.scratch.yaml',  # hyperparameters for the train
        'data_yaml_fname': 'dlp_data.yaml',
        'log_level': 'DEBUG'
    }

    def __init__(self, model_entity):
        super(ModelAdapter, self).__init__(model_entity)
        self._set_device(device_name="cuda:0")
        self.label_map = {}
        self.__add_logger__(level=self.configuration['log_level'])
        self.logger.info('Model Adapter instance created. torch_adapter_v6.0 branch')
        # FIXME: remove create a flow for setting new labels
        #                using single value in the input_shape
        self.logger.debug("Adapter version from 14-Nov-2021")

    # ===============================
    # NEED TO IMPLEMENT THESE METHODS
    # ===============================

    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            This function is called by load_from_snapshot (download to local and then loads)

        :param local_path: `str` directory path in local fileSystem where the weights is taken from
        """
        # get the global paths
        # TODO: should in just verify that i've uploaded from snapshot first?!
        weights_filename = self.snapshot.configuration.get('weights_filename', self.configuration['model_fname'])
        model_path = os.path.join(local_path, weights_filename)
        input_shape = self.snapshot.configuration.get('input_shape', self.configuration['input_shape'])

        self.logger.info("Loading a model from {}".format(local_path))
        # load model arch and state

        # TODO: issues with saving the model with `model.model.model[-1].inplace  (Detect)
        #   See models/experimental.py   - attempt_load function  # 101 for compatability issues
        model = torch.load(model_path, map_location=self.device)
        if isinstance(model, dict):
            state_dict = model
            self.model = state_dict['model']
        else:
            self.model = model

        self.model = self.model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        self.model.to(self.device)
        if self.device.type == 'cpu':
            self.model.float()  # Default model is trained with GPU  as a Half tyep tensor

        # load classes
        self.label_map = {k: v for k, v in enumerate(self.model.names if hasattr(self.model, 'names') else self.model.module.names)}
        self.model.to(self.device)  # TODO: TEST THIS, i had some issues with this command in yolo v5 - for GPU
        self.model.eval()

        # How to load the label_map from loaded model
        self.logger.info("Loaded model from {} successfully".format(model_path))

        # # Save the pytorch preprocess
        # self.preprocess = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(input_shape),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ]
        # )

    def predict(self, batch, **kwargs):
        """ Model inference (predictions) on batch of image
        :param batch: `np.ndarray`
        :return `list[dl.AnnotationCollection]` prediction results by len(batch)
        """
        # return self.predict_w_tensor(batch=batch, **kwargs)  # TODO: fix the scales - should run faster...
        return self.predict_w_autoshape(batch=batch, **kwargs)

    def predict_w_tensor(self, batch, **kwargs):
        """ Model inference (predictions) on batch of image
        :param batch: `np.ndarray`
        :return `list[dl.AnnotationCollection]` prediction results by len(batch)
        """

        min_score = kwargs.get('min_score', 0.4)
        img_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.configuration['input_shape']),   # [::-1]
                # Resize expect width height while self.input_shape is in hxw
                # TODO: consider using letter box (with max input shape - as in train) + letterbox works on np images
                transforms.ToTensor(),
                self.halfTransform(self.half),   # uint8 to fp16/32
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # TODO: the resutls are more consistent when not using the normalize
            ]
        )
        letter_transform = transforms.Compose(
            [
                self.letterBoxTransform(new_shape=self.configuration['input_shape'][::-1], verbose=True),   # shape in HW
                transforms.ToTensor(),
                self.halfTransform(self.half),   # uint8 to fp16/32
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # TODO: the resutls are more consistent when not using the normalize
            ]
        )


        # TODO: test if new version support these edgecases:
        #   1) gray scale images -> stack to 3 channesl
        #   2) alpha channel -> removes it
        img_tensors, orig_shapes = [], []
        for img in batch:
            img_tensors.append(img_transform(img.astype('uint8')))
            orig_shapes.append(img.shape[:2])  # NOTE: numpy shape is height, width (rows,cols) while PIL.size is width, height

        batch_tensor = torch.stack(img_tensors).to(self.device)

        # Inference
        self.logger.debug("{n!r} inference, batch shape {s} ({t!r})".format(n=self.model_name, s=batch_tensor.shape, t=batch_tensor.type()))
        result = self.model(batch_tensor, augment=False)
        dets = result[0]
        # Apply NMS
        dets = non_max_suppression(
            dets, min_score, self.configuration['iou_thres'],
            classes=None, agnostic=self.configuration['agnostic_nms']
        )

        predictions = []
        for i in range(len(batch)):
            item_detections = dets[i].detach().cpu().numpy()  # xyxy, conf, class
            item_predictions = ml.predictions_utils.create_collection()
            for idx, (left, top, right, bottom, score, label_id) in enumerate(item_detections):
                self.logger.debug(f"   --Before scaling--                        @ ({top:2.1f}, {left:2.1f}),\t ({bottom:2.1f}, {right:2.1f})")
                top    = round( max(0, np.floor(top + 0.5).astype('int32')), 3)
                left   = round( max(0, np.floor(left + 0.5).astype('int32')), 3)
                bottom = round( min(orig_shapes[i][0], np.floor(bottom + 0.5).astype('int32')), 3)
                right  = round( min(orig_shapes[i][1], np.floor(right + 0.5).astype('int32')), 3)
                label  = self.label_map[int(label_id)]
                self.logger.debug(f"\tBox {idx:2} - {label:20}: {score:1.3f} @ {(top, left)},\t {(bottom, right)}")
                item_predictions = ml.predictions_utils.add_box_prediction(
                    left=left, top=top, right=right, bottom=bottom,
                    score=score, label=label, adapter=self,
                    collection=item_predictions
                )
            predictions.append(item_predictions)

        return predictions

    def predict_w_autoshape(self, batch, **kwargs):
        """ Model inference (predictions) on batch of image
            Uses the autoshape model which runs preprocess automatically based on the input.
            does not uses hand made transforms
        :param batch: `np.ndarray`
        :return `list[dl.AnnotationCollection]` prediction results by len(batch)
        """
        # Autoshape model inference params - explicit ==> from_config ==> default
        config_min_score = self.configuration.get('min_score', 0.25)
        self.model.conf = min_score = kwargs.get('min_score', config_min_score)
        config_iou_thres = self.configuration.get('iou_thres', 0.45)
        self.model.iou = kwargs.get('iou_thr', config_iou_thres)

        self.logger.debug("AutoShape model NMS params updated, min_score {}, iou_thr {}".format(self.model.conf, self.model.iou))
        results = self.model([im for im in batch])
        self.logger.debug("{n!r} inference with autoshape model, batch shape {s}".
                          format(n=self.model_name, s=results.s,))

        predictions = []
        for i in range(len(batch)):
            item_detections = results.pred[i].detach().cpu().numpy()  # xyxy, conf, class
            item_predictions = ml.predictions_utils.create_collection()
            for idx, (left, top, right, bottom, score, label_id) in enumerate(item_detections):
                label = self.label_map[int(label_id)]
                self.logger.debug(f"\tBox {idx:2} - {label:20}: {score:1.3f} @ {(top, left)},\t {(bottom, right)}")
                item_predictions = ml.predictions_utils.add_box_prediction(
                    left=left, top=top, right=right, bottom=bottom,
                    score=score, label=label, adapter=self,
                    collection=item_predictions
                )
            predictions.append(item_predictions)
        return predictions

    def save(self, local_path, **kwargs):
        """
         saves configuration and weights locally

              Virtual method - need to implement

              the function is called in save_to_snapshot which first save locally and then uploads to snapshot entity

          :param local_path: `str` directory path in local FileSystem
        """
        weights_filename = kwargs.get('weights_filename', self.configuration['weights_filename'])
        weights_path = os.path.join(local_path, weights_filename)
        torch.save(self.model, weights_path)
        self.snapshot.configuration['weights_filename'] = weights_filename
        self.snapshot.configuration['label_map'] = self.label_map
        self.snapshot.update()

    def train(self, data_path, output_path, **kwargs):
        """ Train the model according to data in local_path and save the snapshot to dump_path

            Virtual method - need to implement
        :param data_path: `str` local File System path to where the data was downloaded and converted at
        :param output_path: `str` local File System path where to dump training mid-results (checkpoints, logs...)
        """
        # TODO: set the name of the labels to the model
        # Use opt.data....
        import train as train_script
        # configuration = self.configuration
        # configuration.update(self.snapshot.configuration)
        # num_epochs = configuration.get('num_epochs', 10)
        # batch_size = configuration.get('batch_size', 64)

        if os.path.isfile(self.configuration['hyp_yaml_fname']):
            hyp_full_path = self.configuration['hyp_yaml_fname']
        else:
            hyp_full_path = os.path.join(os.path.dirname(__file__), self.configuration['hyp_yaml_fname'])
        hyp = yaml.safe_load(open(hyp_full_path, 'r'))
        opt = self._create_opt(data_path=data_path, output_path=output_path, **kwargs)
        self.logger.info("Created OPT configuration: batch_size {b};  num_epochs {num} image_size {sz}".
                         format(b=opt.batch_size, num=opt.epochs, sz=opt.imgsz))
        self.logger.debug("OPT config full debug: {}".format(opt))
        # Make sure opt.weights has the exact model file as it will load from there

        train_results = train_script.train(hyp, opt, self.device, callbacks=Callbacks())
        self.logger.info('Train Finished. Actual output path: {}'.format(opt.save_dir))

        # load best model weights
        best_model_wts = os.path.join(opt.save_dir, 'weights', 'best.pt')
        self.model = torch.load(best_model_wts, map_location=self.device)['model']
        # self.model.load_state_dict(best_model_wts)

    def convert_from_dtlpy(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param data_path: `str` local File System directory path where we already downloaded the data from dataloop platform
        :return:
        """

        # update the label_map {id: label} to the one from the snapshot
        label_to_id = {v: k for k, v in self.snapshot.label_map.items()}  # snapshot.label_map {id: name}
        self.label_map = self.snapshot.label_map  # TODO test if label map is okay or need to use labels...

        # White / Black list option to use
        white_list = kwargs.get('white_list', False)  # white list is the verified annotations labels to work with
        black_list = kwargs.get('black_list', False)  # black list is the illegal annotations labels to work with
        empty_prob = kwargs.get('empty_prob', 0)  # do we constraint number of empty images

        for partiton in dl.SnapshotPartitionType:
            in_labels_path = os.path.join(data_path, partiton, 'json')
            in_images_path = os.path.join(data_path, partiton, 'items')

            # Train - Val split
            labels_path = os.path.join(data_path, partiton, 'labels')
            images_path = os.path.join(data_path, partiton, 'images')

            # TODO: currently the function is called inside partition loop - need to fix
            if os.path.isdir(labels_path):
                if len(os.listdir(labels_path)) > 0:
                    self.logger.warning('dir {} already been processed. Skipping'.format(labels_path))
                    continue
            else:
                os.makedirs(labels_path, exist_ok=True)
                os.makedirs(images_path, exist_ok=True)

            # set the list of files to parse and convert
            json_filepaths = list()
            for path, subdirs, files in os.walk(in_labels_path):
                # break
                for fname in files:
                    filename, ext = os.path.splitext(fname)
                    if ext.lower() not in ['.json']:
                        continue
                    json_filepaths.append(os.path.join(path, fname))
            np.random.shuffle(json_filepaths)


            counters = {
                'empty_items_found': 0,
                'empty_items_discarded': 0,
                'corrupted_cnt': 0
            }
            pool = ThreadPool(processes=16)
            lock = Lock()
            for in_json_filepath in tqdm.tqdm(json_filepaths, unit='file'):
                pool.apply_async(func=self._parse_single_annotation_file,
                                 args=(in_json_filepath, in_labels_path, labels_path,
                                       in_images_path, images_path, label_to_id, counters, lock),
                                 kwds={'white_list': white_list,
                                       'black_list': black_list,
                                       'empty_prob': empty_prob}
                                 )
            pool.close()
            pool.join()
            pool.terminate()


        config_path = os.path.join(data_path, self.configuration['data_yaml_fname'])
        self.create_yaml(
            train_path=os.path.join(data_path, dl.SnapshotPartitionType.TRAIN),
            val_path=os.path.join(data_path, dl.SnapshotPartitionType.VALIDATION),
            classes=list(label_to_id.keys()),
            config_path=config_path,
        )

        train_cnt = sum([len(files) for r, d, files in os.walk(data_path+'/train/labels')])
        val_cnt = sum([len(files) for r, d, files in os.walk(data_path+'/validation/labels')])

        msg = "Finished converting the data. Creating config file: {!r}. ".format(config_path) + \
              "\nLabels dict {}.\nlabel_map   {}".format(label_to_id, self.label_map) + \
              "\nVal count   : {}\nTrain count: {}".format(val_cnt, train_cnt)
        self.logger.info(msg)

    def _parse_single_annotation_file(self, in_json_filepath, in_labels_path, labels_path,
                                      in_images_path, images_path, label_to_id, counters, lock,
                                      white_list=False, black_list=False, empty_prob=0):
        try:
            # read the item json
            with open(in_json_filepath, 'r') as f:
                data = json.load(f)
            annotations = dl.AnnotationCollection.from_json(_json=data['annotations'])
            if 'itemMetadata' in data:  # support both types of json files
                item_metadata = data['itemMetadata']
            else:
                item_metadata = data['metadata']

            # partition = item_metadata['system']['snapshotPartition']
            img_width, img_height = item_metadata['system']['width'], item_metadata['system']['height']

            output_txt_filepath = in_json_filepath.replace(in_labels_path, labels_path).replace('.json', '.txt')
            os.makedirs(os.path.dirname(output_txt_filepath), exist_ok=True)
            item_lines = list()
            for ann in annotations:
                if ann.type == 'box':

                    # skip annotation if on white / black list
                    if white_list and ann.label not in white_list:
                        continue
                    if black_list and ann.label in black_list:
                        continue

                    a_h = round(ann.bottom - ann.top, 5)
                    a_w = round(ann.right - ann.left, 5)
                    x_c = round(ann.left + (a_w / 2), 5)
                    y_c = round(ann.top + (a_h / 2), 5)
                    label = ann.label
                    with lock:
                        if label not in label_to_id:
                            label_to_id[label] = len(label_to_id)
                    label_id = label_to_id[label]
                    line = '{label_id} {x_center} {y_center} {width} {height}'.format(
                        label_id=label_id, x_center=x_c / img_width, y_center=y_c / img_height,
                        width=a_w / img_width, height=a_h / img_height)
                    item_lines.append(line)

            if len(item_lines) == 0:
                with lock:
                    counters['empty_items_found'] += 1
                if empty_prob > 0 and np.random.random() < empty_prob:  # save empty image with some prob
                    with lock:
                        counters['empty_items_discarded'] += 1
                    return

            # Create new files in the train-set
            dst = images_path + data['filename']
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src=in_images_path + data['filename'], dst=dst)
            with open(output_txt_filepath, 'w') as f:
                f.write('\n'.join(item_lines))
                f.write('\n')
        except Exception:
            with lock:
                counters['corrupted_cnt'] += 1
            self.logger.error("file: {} had problem. Skipping\n\n{}".format(in_json_filepath, traceback.format_exc()))

    def create_yaml(self, train_path, val_path, classes, config_path='/tmp/dlp_data.yaml'):
        """
        Create the data (or is it the config) yaml
        """

        yaml_template = Path('data_yaml_template.txt')
        self.logger.info("DEBUG: Yaml path: {}; full path: {}".format(yaml_template, yaml_template.absolute()))
        self.logger.info("DEBUG: Test rel path: {}".format(Path('.data_yaml_template.txt').absolute()))

        template = Template(yaml_template.open('r').read())
        yaml_str = template.substitute({
            'train_path': train_path,
            'val_path': val_path,
            'nof_classes': len(classes),
            'classes': classes
        })

        with open(config_path, 'w') as f:
            f.write(yaml_str)

    def _create_opt(self, data_path, output_path, **kwargs):
        import argparse
        data_yaml_path = os.path.join(data_path, self.configuration['data_yaml_fname'])
        if kwargs.get('auto_increase', False) and os.path.isdir(output_path):
                output_path = increment_path(Path(output_path)).as_posix()

        parser = argparse.ArgumentParser()
        parser.add_argument('--save_dir',          type=str, default=output_path, help='path to save the results')
        parser.add_argument('--epochs',            type=int, default=self.configuration.get('num_epochs', 100))  # 300
        parser.add_argument('--batch-size',        type=int, default=self.configuration.get('batch_size', 4), help='batch size for all GPUs')
        # parser.add_argument('--total-batch-size',  type=int, default=16, help='total batch size for all GPUs')
        parser.add_argument('--weights',           type=str, default=self.configuration['model_fname'], help='initial weights file name')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=max(self.configuration['input_shape']), help='train, val image size (pixels)')
        parser.add_argument('--data',              type=str, default=data_yaml_path, help='dlp_data.yaml path')

        parser.add_argument('--global_rank',       type=int, default=-1, help='DDP parameter, do not modify')
        parser.add_argument('--local_rank',        type=int, default=-1, help='DDP parameter, do not modify')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')

        parser.add_argument('--evolve',  type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
        parser.add_argument('--noval',   action='store_true', help='only validate final epoch')
        parser.add_argument('--nosave',  action='store_true', help='only save final checkpoint')
        parser.add_argument('--cfg',     type=str,              default='', help='model.yaml path')
        parser.add_argument('--resume',  nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--workers', type=int,              default=8, help='maximum number of dataloader workers')
        parser.add_argument('--freeze',  type=int,              default=0, help='Number of layers to freeze. backbone=10, all=24')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')
        parser.add_argument('--linear-lr', action='store_true', help='linear LR')
        parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
        parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')

        # NEW
        # parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
        # parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        # parser.add_argument('--project', default='runs/train', help='save to project/name')
        # parser.add_argument('--name', default='exp', help='save to project/name')
        # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        # parser.add_argument('--entity', default=None, help='W&B entity')
        # parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
        # parser.add_argument('--bbox_interval', type=int, default=-1,
        #                     help='Set bounding-box image logging interval for W&B')
        # parser.add_argument('--artifact_alias', type=str, default="latest",
        #                     help='version of dataset artifact to be used')

        opt = parser.parse_known_args()
        return opt[0]

    def _set_device(self, device_name='cuda:0'):
        """
        Set the device and half properties for the adapter
        """
        if not torch.cuda.is_available():
            device_name = 'cpu'
        self.device = torch.device(device_name)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

    def __add_logger__(self, level='INFO'):
        """Adds logger to object"""

        self.logger = logging.getLogger(name=self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        fmt = '%(levelname).1s: %(name)-20s %(asctime)-s [%(filename)-s:%(lineno)-d](%(funcName)-s):: %(msg)s'
        hdl = logging.StreamHandler()
        hdl.setFormatter(logging.Formatter(fmt=fmt, datefmt='%X'))
        hdl.setLevel(level=level.upper())
        hdl.name = 'adapter_handler'  # use the name to get the specific logger
        self.logger.addHandler(hdlr=hdl) 

    def _set_adapter_handler(self, level):
        """Changes the adapter handler level"""
        for hdl in self.logger.handlers:
            if hdl.name.startswith('adapter'):
                hdl.setLevel(level=level)

    class halfTransform(object):
        """ preforms tensor.half if the the model uses half tensors"""
        def __init__(self, is_half):
            self.is_half = is_half

        def __call__(self, sample):
            # uint8 to fp16/32
            if self.is_half:
                return sample.half()
            else:
                return sample.float()

    class letterBoxTransform(object):
        """ preforms letterbox reshape - works on numpy inputs ==> HW"""
        def __init__(self, new_shape, verbose=False):
            self.new_shape = new_shape
            self.verobse = verbose

        def __call__(self, img: np.ndarray):
            im, ratio, (dw, dh) = letterbox(im=img,
                                            new_shape=self.new_shape,
                                            )
            if self.verbose:
                print(f"letter box reshapes {img.shape} -> {im.shape} (ratio: {ratio},  dw:{dw}-dh{dh})")
            return im



def _get_coco_labels_json():
    return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']  # class names


def model_and_snapshot_creation(env='prod', yolo_size='small'):
    dl.setenv(env)
    project = dl.projects.get('DataloopModels')

    model = model_creation(env=env, project=project)
    snapshot = snapshot_creation(model, env=env, yolo_size=yolo_size)


def model_creation(env='prod', project:dl.Project = None):
    dl.setenv(env)

    if project is None:
        project = dl.projects.get('DataloopModels')

    codebase = dl.GitCodebase(git_url='https://github.com/dataloop-ai/yolov5.git', git_tag='torch_adapter_v6.0') 
    model = project.models.create(model_name='yolo-v5',
                                  description='Global Dataloop Yolo V5 implemented in pytorch',
                                  output_type=dl.AnnotationType.BOX,
                                  is_global= (project.name == 'DataloopModels'),
                                  tags=['torch', 'yolo', 'detection'],
                                  codebase=codebase,
                                  entry_point='model_adapter.py',
                                  )
    return model


def snapshot_creation(model, env='prod', yolo_size='small'):
    dl.setenv(env)
    # TODO: can we add two model arc in one dir - yolov5l, yolov5s

    # Select the specific arch and gcs bucket
    if yolo_size == 'small':
        gcs_prefix = 'yolo-v5-v6/small'
        abbv = 's'
    elif yolo_size == 'large':
        gcs_prefix = 'yolo-v5-v6/large'
        abbv = 'l'
    elif yolo_size == 'extra':
        gcs_prefix = 'yolo-v5-v6/extra'
        abbv = 'x'
    else:
        raise RuntimeError('yolo_size {!r} - un-supported, choose "small" "large" or "extra" '.format(yolo_size))


    bucket = dl.buckets.create(dl.BucketType.GCS,
                               gcs_project_name='viewo-main',
                               gcs_bucket_name='model-mgmt-snapshots',
                               gcs_prefix=gcs_prefix)
    snapshot = model.snapshots.create(snapshot_name='pretrained-yolo-v5-{}'.format(yolo_size),
                                      description='yolo v5 {} arch, pretrained on ms-coco'.format(yolo_size),
                                      tags=['pretrained', 'ms-coco'],
                                      dataset_id=None,
                                      is_global=model.is_global,
                                      status='trained',
                                      configuration={'weights_filename': 'yolov5{}.pt'.format(abbv),
                                                     # 'classes_filename': 'classes.json'
                                                     },
                                      project_id=model.project.id,
                                      bucket=bucket,
                                      labels=_get_coco_labels_json()
                                      )
    return snapshot
