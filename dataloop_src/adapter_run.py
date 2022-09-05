# import dtlpy as dl
# import json
#
# dl.setenv('dev')
# dataset = dl.datasets.get(None, '62d8e4e08881be03fec87fed')
#
# dataset.metadata['system'] = {"subsets": {"train": json.dumps(dl.Filters().prepare()),
#                                           "validation": json.dumps(dl.Filters().prepare()),
#                                           "test": None}}
# dataset.update(True)


from dataloop_src.model_adapter import ModelAdapter
import dtlpy as dl
import numpy as np
import json

dl.setenv('dev')
model_entity = dl.models.get(None, '630df1a34a273a156144019b')
package = model_entity.package
# model_entity.bucket.upload(r"C:\Users\Shabtay\Downloads\New folder")
adapter = ModelAdapter(None)
adapter.load_from_model(model_entity=model_entity)


def prd():
    item = dl.items.get(None, '630cd62e7a00b0b71a95196c')
    a = adapter.predict_items(items=[item], with_upload=True)


def cln():
    to_dataset = dl.datasets.get(None, '630caf79ee6c90226a406f31')
    to_project = to_dataset.project

    m = model_entity.clone(model_name='fruits with figures exp2',
                           description='exp 2',
                           # labels=list(to_dataset.labels_flat_dict.keys()),
                           dataset=to_dataset,
                           configuration={'num_epochs': 20,
                                          'batch_size': 4},
                           project_id=to_project.id)


# cln()
def trn():
    package.models.list().print()
    m = dl.models.get(model_id='6310845dca2e52b041de5f88')
    # m.labels = list(m.dataset.labels_flat_dict.keys())
    # m.update()

    # split
    # items = list(m.dataset.items.list().all())
    # print(np.unique([item.dir for item in items]))
    #
    # item: dl.Item
    # for item in items:
    #     if np.random.random() > 0.8:
    #         item.move(f'/train{item.filename}')
    #     else:
    #         item.move(f'/validation{item.filename}')
    # m.dataset.metadata['system']['subsets'] = {
    #     'train': json.dumps(dl.Filters(field='dir', values='/train').prepare()),
    #     'validation': json.dumps(dl.Filters(field='dir', values='/test').prepare()),
    # }
    # m.dataset.update(True)
    adapter.train_model(m)


def trn_service():
    to_dataset = dl.datasets.get(None, '62e000ad95c4f602ae9691c8')
    to_project = dl.projects.get(None, 'a179220f-bd40-4a75-97e3-fbe05c9f276f')

    m = model_entity.clone(model_name='new sheeps exp',
                           description='exp 3',
                           labels=list(to_dataset.labels_flat_dict.keys()),
                           dataset_id=to_dataset.id,
                           project_id=to_project.id)

    to_project.models.list().print()
    to_project.packages.list().print()
    m = to_project.models.get(model_name='new sheeps exp2')

    sec, res = dl.client_api.gen_request("post",
                                         f"/ml/models/{m.id}/train")


def upload_metric():
    m = dl.models.get(model_id='62e1435336ae84d7b58a8cfa')

    # m.add_log_samples(samples=dl.LogSample(figure="loss",
    #                                        legend="train",
    #                                        x=0,
    #                                        y=1),
    #                   dataset_id=m.dataset_id)

    x = np.arange(1, 51)
    loss = np.exp(-x / 2)
    acc = np.log(x / 10)
    acc -= np.min(acc)
    acc /= np.max(acc)

    train_loss = list()
    train_acc = list()
    val_loss = list()
    val_acc = list()
    for i_epoch in range(0, 50):
        train_loss.append(loss[i_epoch] + np.random.random() / 10)
        train_acc.append(acc[i_epoch] + np.random.random() / 10)
        val_loss.append(loss[i_epoch] + np.random.random() / 10)
        val_acc.append(acc[i_epoch] + np.random.random() / 10)

        # train
        sample = dl.LogSample(figure='train',
                              legend='loss',
                              x=i_epoch,
                              y=loss[i_epoch] + np.random.random() / 10)
        m.add_log_samples(sample, dataset_id=m.dataset_id)
        sample = dl.LogSample(figure='train',
                              legend='accuracy',
                              x=i_epoch,
                              y=acc[i_epoch] + np.random.random() / 10)
        m.add_log_samples(sample, dataset_id=m.dataset_id)
        # val
        sample = dl.LogSample(figure='val',
                              legend='loss',
                              x=i_epoch,
                              y=loss[i_epoch] + np.random.random() / 10)
        m.add_log_samples(sample, dataset_id=m.dataset_id)
        sample = dl.LogSample(figure='val',
                              legend='accuracy',
                              x=i_epoch,
                              y=acc[i_epoch] + np.random.random() / 10)
        m.add_log_samples(sample, dataset_id=m.dataset_id)
        import time

        time.sleep(3)


if __name__ == "__main__":
    trn()
