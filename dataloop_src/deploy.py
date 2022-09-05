import dtlpy as dl
import cv2
import os


def get_coco_labels_json():
    return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']  # class names


def model_and_snapshot_creation(env='prod', yolo_size='small'):
    env = 'rc'
    yolo_size = 'openvino'
    dl.setenv(env)
    project = dl.projects.get('DataloopModels')
    model = model_creation(env=env, project=project)
    snapshot = snapshot_creation(model, env=env, yolo_size=yolo_size)


def model_creation(env='prod', project: dl.Project = None):
    dl.setenv(env)
    if project is None:
        project = dl.projects.get('DataloopModels')

    codebase = dl.GitCodebase(git_url='https://github.com/dataloop-ai/yolov5.git',
                              git_tag='dtlpy-v6.1.1')
    model = project.models.create(model_name='yolo-v5',
                                  description='Global Dataloop Yolo V5 implemented in pytorch',
                                  output_type=dl.AnnotationType.BOX,
                                  is_global=(project.name == 'DataloopModels'),
                                  tags=['torch', 'yolo', 'detection'],
                                  codebase=codebase,
                                  entry_point='model_adapter.py',
                                  default_runtime=dl.KubernetesRuntime(runner_image=''),
                                  default_configuration={'size': 640})
    return model


def snapshot_creation(model, env='prod', yolo_size='small'):
    env = 'rc'
    yolo_size = 'openvino'
    dl.setenv(env)
    # TODO: can we add two model arc in one dir - yolov5l, yolov5s

    # Select the specific arch and gcs bucket
    if yolo_size == 'small':
        gcs_prefix = 'yolo-v5-v6/small'
        weights_filename = 'yolov5s.pt'
    elif yolo_size == 'large':
        gcs_prefix = 'yolo-v5-v6/large'
        weights_filename = 'yolov5l.pt'
    elif yolo_size == 'extra':
        gcs_prefix = 'yolo-v5-v6/extra'
        weights_filename = 'yolov5x.pt'
    elif yolo_size == 'openvino':
        gcs_prefix = 'yolo-v5-v6/openvino'
        weights_filename = 'yolov5s.xml'


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
                                      configuration={'weights_filename': weights_filename,
                                                     'img_size': [640, 640],
                                                     'conf_thres': 0.25,
                                                     'iou_thres': 0.45,
                                                     'max_det': 1000,
                                                     'device': 'cuda',
                                                     'agnostic_nms': False,
                                                     'half': False,
                                                     'id_to_label_map': {ind: label for ind, label in
                                                                         enumerate(get_coco_labels_json())}},
                                      project_id=model.project.id,
                                      bucket=bucket,
                                      labels=get_coco_labels_json()
                                      )
    return snapshot


def deploy_service():
    import os
    import dtlpy as dl
    func = [dl.PackageFunction(name='run',
                               inputs=[dl.FunctionIO(type="Item", name="item"),
                                       dl.FunctionIO(type="Json", name="config")],
                               description='Inference on pre-trained YOLO V5'
                               )
            ]
    modules = [dl.PackageModule(entry_point='model_service.py',
                                init_inputs=[
                                    dl.FunctionIO(name='model_id',
                                                  type=dl.PACKAGE_INPUT_TYPE_JSON),
                                    dl.FunctionIO(name='snapshot_id',
                                                  type=dl.PACKAGE_INPUT_TYPE_JSON)
                                ],
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

    env = 'rc'
    dl.setenv(env)

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
    models_project = dl.projects.get('DataloopModels')
    model = models_project.models.get('yolo-v5')
    snapshot = model.snapshots.get('pretrained-yolo-v5-openvino')
    service = package.services.deploy(service_name=package_name,
                                      init_input=[
                                          dl.FunctionIO(name='model_id',
                                                        type=dl.PACKAGE_INPUT_TYPE_JSON,
                                                        value=model.id),
                                          dl.FunctionIO(name='snapshot_id',
                                                        type=dl.PACKAGE_INPUT_TYPE_JSON,
                                                        value=snapshot.id)
                                      ],
                                      runtime={'gpu': False,
                                               'numReplicas': 1,
                                               'podType': 'regular-s',
                                               'concurrency': 20,
                                               'runnerImage': 'dataloop_runner-cpu/yolov5-openvino:1'},
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

    def execute():
        service = dl.services.get(service_name='model-annotation-yolov5')
        service.execute(execution_input=[dl.FunctionIO(type='Item',
                                                       name='item',
                                                       value='61eac3fc35e2fd9ce2f148ee'),
                                         dl.FunctionIO(type='Json',
                                                       name='config',
                                                       value={'output_action': 'draw'})])


def remote_model_test():
    from model_adapter import ModelAdapter
    dl.setenv('rc')
    project = dl.projects.get('COCO ors')
    # model = model_creation('rc', project)
    model = project.models.get('yolo-v5')
    self = ModelAdapter(model_entity=model)
    snapshot = model.snapshots.get('pretrained-yolo-v5-openvino')
    self.load_from_snapshot(snapshot=snapshot)
    item = dl.items.get(item_id='61eac3fc35e2fd9ce2f148ee')
    batch_annotations = self.predict_items(items=[item])


def remote_service_test():
    from model_service import ServiceRunner
    dl.setenv('rc')
    project = dl.projects.get('COCO ors')
    # model = model_creation('rc', project)
    model = project.models.get('yolo-v5')
    snapshot = model.snapshots.get('pretrained-yolo-v5-openvino')

    self = ServiceRunner(dl=dl, model_id=model.id, snapshot_id=snapshot.id)

    item = dl.items.get(item_id='61eac3fc35e2fd9ce2f148ee')
    batch_annotations = self.run(dl=dl, item=item)
    item.annotations.upload(batch_annotations[0])


def local_test():
    from model_adapter import ModelAdapter
    dl.setenv('rc')
    project = dl.projects.get('COCO ors')
    # model = model_creation('rc', project)
    model = project.models.get('yolo-v5')
    self = ModelAdapter(model_entity=model)
    self.configuration['weights'] = 'yolov5s.pt'
    self.configuration['weights'] = 'yolov5s_openvino_model_640_640/yolov5s.xml'
    self.configuration['img_size'] = [640, 640]
    # self.configuration['img_size'] = [640, 480]
    local_path = os.getcwd()
    self.load(local_path=local_path)
    batch = cv2.imread(r'images/000000000064.jpg')[None, ...]

    item = dl.items.get(item_id='61eac3fc35e2fd9ce2f148ee')
    batch_annotations = self.predict(batch=batch)
    item.annotations.upload(batch_annotations[0])

    #
    #
    # item = dl.items.get(item_id='61eac3fc35e2fd9ce2f148ee')
    # item.annotations.upload(batch_annotations[0])
    # self.predict_items(items=[])


if __name__ == "__main__":
    remote_service_test()
    # remote_model_test()
