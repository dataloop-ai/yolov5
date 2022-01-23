import time
import dtlpy as dl


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self, dl, model_id, snapshot_id):
        model_entity = dl.models.get(model_id=model_id)
        snapshot_entity = dl.snapshots.get(snapshot_id=snapshot_id)
        self.adapter = model_entity.build()
        self.adapter.load_from_snapshot(snapshot=snapshot_entity)

    def run(self, dl, item, progress=None, config=None):
        if config is None:
            config = dict()
        if 'annotation_type' not in config:
            config['annotation_type'] = 'binary'
        if 'confidence_th' not in config:
            config['confidence_th'] = 0.50
        if 'output_action' not in config:
            config['output_action'] = 'annotations'
        if progress is not None:
            progress.logger.info('input config: %s' % config)

        tic = time.time()
        batch_annotations = self.adapter.predict_items(items=[item])
        builder = batch_annotations[0]
        if config['output_action'] == 'dict':
            annotation_batch = [{'label': ann.label,
                                 'coordinates': ann.coordinates}
                                for ann in builder.annotations]
        elif config['output_action'] == 'annotations':
            annotation_batch = [ann.to_json()
                                for ann in builder.annotations]
        else:
            raise ValueError('unknown output_action in config: %s' % config['output_action'])
        return annotation_batch
