import dtlpy as dl
from dtlpy import ml
import os
import shutil
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
# from torchvision.transforms.functional import
from PIL import Image
import tqdm
import json
import time
from pathlib import Path
import yaml
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock
import traceback
from utils.general import increment_path, non_max_suppression
from utils.callbacks import Callbacks



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

    }
    _defaults = {
        'input_shape': (480, 640),  # height, width - numpy format
        'config_name':  'config.yaml',
        'weights_filename': 'yolov5l.pt',
        'weights_path': os.path.join(os.path.expandvars('$ZOO_CONFIGS'), 'yolov5_torch', 'base', 'yolov5x.pt'),
        'conf_thres': 0.4,  # help='object confidence threshold')
        'iou_thres': 0.5,  # help='IOU threshold for NMS')
        'fourcc': 'mp4v',  # help='output video codec (verify ffmpeg support)')
        'device_name': 0,  # help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        'classes': None,  # , help='filter by class')
        'agnostic_nms': False,  # help='class-agnostic NMS')
        'augment': False,  # help='augmented inference')
        'config_deepsort': "deep_sort_pytorch/configs/deep_sort.yaml",

    }

    def __init__(self, model_entity):
        super(ModelAdapter, self).__init__(model_entity)
        self._set_device(device_name="cuda:0")
        self.label_map = {}
        self.logger.info('Model Adapter instance created. torch_adapter_v6.0 branch')
        # FIXME: remove _defaults, create a flow for setting new labels, tackle the 'inplace' inconsistency
        self.logger.info("This version is Newer than 13-Oct-2021")

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

        min_score = kwargs.get('min_score', 0.4)
        img_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.configuration['input_shape'][::-1]),
                # Resize expect width height while self.input_shape is in hxw
                transforms.ToTensor(),
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
        # TODO: add in the compose - with conditional
        batch_tensor = batch_tensor.half() if self.half else batch_tensor.float()  # uint8 to fp16/32

        # Inference
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
        configuration = self.configuration
        configuration.update(self.snapshot.configuration)
        num_epochs = configuration.get('num_epochs', 10)
        batch_size = configuration.get('batch_size', 64)

        if os.path.isfile(self.configuration['hyp_yaml_fname']):
            hyp_full_path = self.configuration['hyp_yaml_fname']
        else:
            hyp_full_path = os.path.join(os.path.dirname(__file__), self.configuration['hyp_yaml_fname'])
        hyp = yaml.safe_load(open(hyp_full_path, 'r'))
        opt = self._create_opt(data_path=data_path, output_path=output_path, **kwargs)
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
        label_to_id = {v: k for k, v in self.label_map.items()}  # self.label_map {id: name}
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

    def load_old__(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            This function is called by load_from_snapshot (download to local and then loads)

        :param local_path: unused - weights path is taken from self.weights_path
        """
        if not issubclass(self.__class__, dl.BaseModelAdapter):
        #if issubclass(self.__class__, dl.SuperModelAdapter):
            # init grids
            # FIXME : specificed numbers used in V project
            pad_top, pad_left, pad_bottom, pad_right = [194, 386, 2129, 4241]
            self.set_grids_by_nof_boxes(size_wh=(4628, 2324), nof_rows=2, nof_cols=2)

        # Initialize
        if not torch.cuda.is_available():
            self.device_name = 'cpu'
        self.device = torch.device(self.device_name)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        if self.snapshot.configuration.get('use_pretrained', False):
            model_arch = self.weights_filename.split('.')[0]
            model = torch.hub.load('ultralytics/yolov5', model_arch, pretrained=True)
            # FIXME: in case we choose CPU - set all inner tensors to CPU  - not working when torch.cuda.is_avilable()
            # Move entire model to device, as it contatins some inner tensors
            # _ = [st.to(self.device) for st in  model.model.stride]
            # _ = [st.to(self.device) for st in  model.model.model[-1].stride]
            # _ = [gr.to(self.device) for gr in  model.model.model[-1].grid]
            model.to(self.device)

            self.class_map = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

        else:  # Load from file
            if os.path.isfile(self.weights_path):
                weights_path = self.weights_path
            else:
                weights_path = os.path.join(self.weights_path, self.weights_filename)

            model = torch.load(weights_path, map_location=self.device)['model'].float()  # load to FP32
            model.to(self.device).eval()

            self.class_map = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
            if self.half:
                model.half()  # to FP16
            self.logger.info("Model loaded from  {}. Other attrs: {}".
                             format(weights_path, {k: self.__getattribute__(k) for k in self._defaults.keys()}))

            # If on GPU -  run with empty image
            if self.device_name != 'cpu':
                zeroes_img = torch.zeros((1, 3, self.input_shape[0], self.input_shape[1]), device=self.device)  # init img
                if self.half:
                    zeroes_img = zeroes_img.half()
                _ = model(zeroes_img)

        self.model = model

    def train_old___(self, data_path, dump_path, **kwargs):
        """ Train the model according to data in local_path and save the snapshot to dump_path
        :param data_path: `str` local File System path to where the data was downloaded and converted at
        :param dump_path: `str` local File System path where to dump training mid-results (checkpoints, logs...)
        """
        import train as train_script
        if os.path.isfile(self.hyp_yaml_fname):
            hyp_full_path = self.hyp_yaml_fname
        else:
            hyp_full_path = os.path.join(os.path.dirname(__file__), self.hyp_yaml_fname)
        hyp = yaml.safe_load(open(hyp_full_path, 'r'))
        opt = self._create_opt(data_path=data_path, dump_path=dump_path, **kwargs)
        # Make sure opt.weights has the exact model file as it will load from there

        if self.device is None:
            if not torch.cuda.is_available():
                self.device_name = 'cpu'
            self.device = torch.device(self.device_name)

        results = train_script.train(hyp, opt, self.device)

        # Train scripts returns some results.  We need to load the adapter to the latest state
        act_dump_path = opt.save_dir
        if act_dump_path != dump_path:
            self.logger.warning("Dump path was incremented to {}".format(act_dump_path))
        torch.load(os.path.join(act_dump_path, 'weights', 'best.pt'))
        self.opt = opt
        self.logger.debug("\nUsed opt: \n{}".format(opt))

        self.logger.debug("Use train.py as script for more options during the train")

    def predict_old__(self, batch, verbose=True):
        """ Model inference (predictions) on batch of image
        :param batch: `np.ndarray`
        :return `list[dl.AnnotationCollection]` prediction results by len(batch)
        """
        from utils.datasets import letterbox
        from utils.general import check_img_size, non_max_suppression, scale_coords

        scaled_batch, orig_shapes = [], []
        for img in batch:
            if len(img.shape) == 2:  # Gray scale image -> expand to 3 channels
                img = np.stack((img,)*3, axis=-1)
            elif img.shape[2] == 4:  # with alpha channel -> remove it
                img = img[:, :, :3]

            orig_shapes.append(img.shape[:2])  # NOTE: numpy shape is height, width (rows,cols) while PIL.size is width, height
            img_scaled = cv2.resize(img, self.input_shape[::-1])  # dsize is width height while self.input_shape is in hxw - np format
            # crop_np, ratio, pad = letterbox(crop_np, (self.nn_shape_h, self.nn_shape_w))  # output is width height
            img_scaled = img_scaled.transpose(2, 0, 1)  #  RGB X height X width
            img_scaled = np.ascontiguousarray(img_scaled)
            scaled_batch.append(img_scaled)

        scaled_batch = np.array(scaled_batch)

        batch_torch = torch.from_numpy(scaled_batch).to(self.device)
        batch_torch = batch_torch.half() if self.half else batch_torch.float()  # uint8 to fp16/32
        batch_torch /= 255.0  # 0 - 255 to 0.0 - 1.0
        if batch_torch.ndimension() == 3:
            batch_torch = img.unsqueeze(0)

        # Inference
        t1 = time.time()   # time_synchronized()
        result = self.model(batch_torch, augment=self.augment)
        dets = result[0]
        # Apply NMS
        dets = non_max_suppression(
            dets, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms
        )

        predictions = []
        for i in range(len(batch)):
            item_detections = dets[i].detach().cpu().numpy()  # xyxy, conf, class
            nof_detections = len(item_detections)
            item_predictions = ml.predictions_utils.create_collection()
            for b in range(nof_detections):
                scale_h, scale_w = np.array(orig_shapes[i]) / np.array(self.input_shape)
                left, top, right, bottom, score, label_id = item_detections[b]
                self.logger.debug(f"   --Before scaling--                        @ ({top:2.1f}, {left:2.1f}),\t ({bottom:2.1f}, {right:2.1f})")
                top    = round( max(0, np.floor(top + 0.5).astype('int32')) * scale_h, 3)
                left   = round( max(0, np.floor(left + 0.5).astype('int32')) * scale_w, 3)
                bottom = round( min(orig_shapes[i][0], np.floor(bottom + 0.5).astype('int32') * scale_h), 3)
                right  = round( min(orig_shapes[i][1], np.floor(right + 0.5).astype('int32') * scale_w), 3)
                label  = self.class_map[int(label_id)]
                self.logger.debug(f"\tBox {b:2} - {label:20}: {score:1.3f} @ {(top, left)},\t {(bottom, right)}")
                item_predictions = ml.predictions_utils.add_box_prediction(
                    left=left, top=top, right=right, bottom=bottom,
                    score=score, label=label, adapter=self,
                    collection=item_predictions
                )
            predictions.append(item_predictions)

        return predictions

    def convert_old__(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            e.g. take dlp dir structure and construct annotation file

        :param data_path: `str` local File System directory path where we already downloaded the data from dataloop platform
        :return: creates a txt file for each image file we have.
            txt file has lines of all the labels.
            each line is in format :{label_id} {x_center} {y_center} {width} {height} all normalized for the image shape
        """

        # OPTIONAL FIELDS - KWARGS
        val_ratio = kwargs.get('val_ratio', 0.2)
        # White / Black list option to use
        white_list = kwargs.get('white_list', False)  # white list is the verified annotations labels to work with
        black_list = kwargs.get('black_list', False)  # black list is the illegal annotations labels to work with
        empty_prob = kwargs.get('empty_prob', 0)  # do we constraint number of empty images
        dir_prefix = kwargs.get('dir_prefix', '')  # prefix dir to separate multiple trains

        # organize filesystem and structure
        # =================================
        in_images_path = os.path.join(data_path, 'items')
        in_labels_path = os.path.join(data_path, 'json')
        # TODO: Test if the dataloader support that the images are not in the train / val directiries
        train_path = os.path.join(data_path, dir_prefix, 'train')
        val_path = os.path.join(data_path, dir_prefix, 'val')

        json_filepaths = list()
        for path, subdirs, files in os.walk(in_labels_path):
            # break
            for fname in files:
                filename, ext = os.path.splitext(fname)
                if ext.lower() not in ['.json']:
                    continue
                json_filepaths.append(os.path.join(path, fname))
        np.random.shuffle(json_filepaths)

        label_to_id = dict()
        self.logger.debug("Preparing the images (#{}) for train: {!r} and Val {!r}. (ratio is set to: {})".
                          format(len(json_filepaths), train_path, val_path, val_ratio))
        # COUNTERS
        counters = {
            'empty_items_found': 0,
            'empty_items_discarded': 0,
            'corrupted_cnt': 0
        }
        pool = ThreadPool(processes=16)
        lock = Lock()
        for in_json_filepath in tqdm.tqdm(json_filepaths, unit='file'):
            # Train - Val split
            if np.random.random() < val_ratio:
                labels_path = os.path.join(val_path, 'labels')
                images_path = os.path.join(val_path, 'images')
            else:
                labels_path = os.path.join(train_path, 'labels')
                images_path = os.path.join(train_path, 'images')

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
        # TODO: counters do not work in new version

        # COUNTERS
        empty_items_found_cnt, empty_items_discarded = counters['empty_items_found'], counters['empty_items_discarded']
        corrupted_cnt = counters['corrupted_cnt']
        actual_empties = empty_items_found_cnt - empty_items_discarded
        train_cnt = sum([len(files) for r, d, files in os.walk(train_path+'/labels')])
        val_cnt = sum([len(files) for r, d, files in os.walk(val_path+'/labels')])

        config_path = os.path.join(data_path, dir_prefix, self.data_yaml_fname)
        msg = "Finished converting the data. Creating config file: {!r}. ".format(config_path) + \
            "\nLabels dict {}.  Found {} empty items".format(label_to_id, empty_items_found_cnt) + \
            "\nVal count   : {}\nTrain count: {}\n(out of them {} empty,  {} corrupted)".\
                  format(val_cnt, train_cnt, actual_empties, corrupted_cnt)

        self.logger.info(msg)
        self.create_yaml(train_path=train_path, val_path=val_path, classes=list(label_to_id.keys()),
                         config_path=config_path)

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

        yaml_str = f"""
    # Train command: python train.py --data {config_path}/dlp_data.yaml
    # Default dataset location is on ~/DATA folder

    # download command/URL (optional)
    # download: Not implemented

    # train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
    train: {train_path}  
    val: {val_path}

    # number of classes
    nc: {len(classes)}

    # class names
    names: {classes}
        """
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

    codebase = dl.GitCodebase(git_url='https://github.com/dataloop-ai/yolov5.git', git_tag='torch_adapter_v6.0') #  TODO:  git_tag='master') 'v5.0' or 6.0
    model = project.models.create(model_name='yolo-v5',
                                  description='Global Dataloop Yolo V5 implemented in pytorch',
                                  output_type=dl.AnnotationType.BOX,
                                  is_global=False,  # FIXME
                                  tags=['torch', 'yolo', 'detection'],
                                  codebase=codebase,
                                  entry_point='model_adapter.py',
                                  )

    # TODO: can we add two model arc in one dir - yolov5l, yolov5s
    # Select the specific arch and gcs bucket
    if yolo_size == 'small':
        gcs_prefix = 'yolo-v5-small'
    elif yolo_size == 'large':
        gcs_prefix = 'yolo-v5'
    else:
        raise RuntimeError('yolo_size {!r} - un-supported, choose "small" or "large"'.format(yolo_size))

    abbv = yolo_size[0]

    bucket = dl.buckets.create(dl.BucketType.GCS,
                               gcs_project_name='viewo-main',
                               gcs_bucket_name='model-mgmt-snapshots',
                               gcs_prefix=gcs_prefix)
    snapshot = model.snapshots.create(snapshot_name='pretrained-yolo-v5',
                                      description='yolo v5 {} arch, pretrained on ms-coco'.format(yolo_size),
                                      tags=['pretrained', 'ms-coco'],
                                      dataset_id=None,
                                      # is_global=True,
                                      # status='trained',
                                      configuration={'weights_filename': 'yolov5{}.pt'.format(abbv),
                                                     # 'classes_filename': 'classes.json'
                                                     },
                                      project_id=project.id,
                                      bucket=bucket,
                                      labels=_get_coco_labels_json()
                                      )
    return snapshot
