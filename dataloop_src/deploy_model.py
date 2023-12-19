import dtlpy as dl
from model_adapter import ModelAdapter
import os
import yaml


def package_creation(project: dl.Project):
    metadata = dl.Package.get_ml_metadata(cls=ModelAdapter,
                                          default_configuration={'weights_filename': 'yolov5s.pt',
                                                                 'num_epochs': 10,
                                                                 'batch_size': 4,
                                                                 'img_size': [640, 640],
                                                                 'conf_thres': 0.25,
                                                                 'iou_thres': 0.45,
                                                                 'max_det': 1000,
                                                                 'device': 'cuda',
                                                                 'agnostic_nms': False,
                                                                 'half': False},
                                          output_type=dl.AnnotationType.BOX,
                                          )
    modules = dl.PackageModule.from_entry_point(entry_point='dataloop_src/model_adapter.py')

    package = project.packages.push(package_name='yolov5',
                                    src_path=os.getcwd(),
                                    # description='Global Dataloop Yolo V5 implemented in pytorch',
                                    is_global=True,
                                    package_type='ml',
                                    codebase=dl.GitCodebase(git_url='https://github.com/dataloop-ai/yolov5.git',
                                                            git_tag='mgmt3'),
                                    modules=[modules],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_GPU_K80_S,
                                                                        runner_image='gcr.io/viewo-g/piper/agent/runner/gpu/yolov5-openvino-gpu:3',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        concurrency=1).to_json(),
                                        'initParams': {'model_entity': None}
                                    },
                                    metadata=metadata)
    package.build()
    return package


def model_creation(package: dl.Package, yolo_size='small'):
    # TODO: can we add two model arc in one dir - yolov5l, yolov5s
    # Select the specific arch and gcs bucket
    if yolo_size == 'small':
        url = 'https://storage.googleapis.com/model-mgmt-snapshots/yolo-v5-v6/small/yolov5s6.pt'
        weights_filename = 'yolov5s6.pt'
    elif yolo_size == 'large':
        url = 'https://storage.googleapis.com/model-mgmt-snapshots/yolo-v5-v6/large/yolov5l.pt'
        weights_filename = 'yolov5l.pt'
    elif yolo_size == 'extra':
        url = 'https://storage.googleapis.com/model-mgmt-snapshots/yolo-v5-v6/extra/yolov5x.pt'
        weights_filename = 'yolov5x.pt'
    else:
        raise RuntimeError('yolo_size {!r} - un-supported, choose "small" "large" or "extra" '.format(yolo_size))
    with open('data/coco.yaml', encoding='utf=8') as f:
        coco_yaml = yaml.safe_load(f)
    labels = coco_yaml['names']

    model = package.models.create(model_name='pretrained-yolo-v5-{}'.format(yolo_size),
                                  description='yolo v5 {} arch, pretrained on ms-coco'.format(yolo_size),
                                  tags=['pretrained', 'ms-coco'],
                                  dataset_id=None,
                                  status='trained',
                                  scope='public',
                                  configuration={
                                      'weights_filename': weights_filename,
                                      'img_size': [640, 640],
                                      'conf_thres': 0.25,
                                      'iou_thres': 0.45,
                                      'max_det': 1000,
                                      'device': 'cuda',
                                      'agnostic_nms': False,
                                      'half': False,
                                      'data_yaml_fname': 'coco.yaml',
                                      'hyp_yaml_fname': 'hyp.finetune.yaml',
                                      'id_to_label_map': {ind: label for ind, label in enumerate(labels)}},
                                  project_id=package.project.id,
                                  model_artifacts=[dl.LinkArtifact(url=url,
                                                                   filename=weights_filename)],
                                  labels=labels
                                  )
    return model


if __name__ == '__main__':
    project_id = ''
    project = dl.projects.get(project_id=project_id)
    package = package_creation(project=project)
    model = model_creation(package=package, yolo_size='small')
    print('model created: {}'.format(model.id))
