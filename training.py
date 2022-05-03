import sys

import PIL
import io
import os
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from torchvision.transforms import ToTensor
import torchaudio
import torch
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random

from clearml import Task, Dataset

import global_config

task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name='training',
    reuse_last_task_id=False
)

configuration_dict = {
    'dropout': 0.15,
    'base_lr': 0.002,
    'number_of_epochs': 10,
    'training_batch_size': 64,
    'testing_batch_size': 64,
    'dataset_id': '6b7fa258482c4dcd9262e62f41af6c98',
    'seed': 1337
}
task.connect(configuration_dict)

# Set ALL the necessary seeds AND if using GPU we need to set it to use deterministic algorithms
random.seed(configuration_dict['seed'])
np.random.seed(configuration_dict['seed'])
torch.manual_seed(configuration_dict['seed'])
torch.cuda.manual_seed_all(configuration_dict['seed'])
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
torch.use_deterministic_algorithms(True)


class ClearMLDataSet(Dataset):
    def __init__(self, folder_filter):
        clearml_dataset = Dataset.get(configuration_dict['dataset_id'])
        task.add_tags(clearml_dataset.tags)
        self.img_dir = clearml_dataset.get_local_copy()
        self.img_metadata = Task.get_task(task_id=clearml_dataset.id).artifacts['metadata'].get()
        self.img_metadata = self.img_metadata[self.img_metadata['fold'].isin(folder_filter)]
        # We just removed some rows by filtering on class, this will make gaps in the dataframe index
        # (e.g. 57 won't exist anymore) so we reindex to make it a full range again, otherwise we'll get errors later
        # when selecting a row by index
        self.img_metadata = self.img_metadata.reset_index(drop=True)

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        sound_path = os.path.join(self.img_dir, self.img_metadata.loc[idx, 'filepath'])
        img_path = sound_path.replace('.wav', '.npy')
        image = np.load(img_path)
        label = self.img_metadata.loc[idx, 'label']
        return sound_path, image, label


train_dataset = ClearMLDataSet(set(range(1, 10)))
test_dataset = ClearMLDataSet({10})
print(len(train_dataset), len(test_dataset))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configuration_dict.get('training_batch_size', 4),
                                           shuffle=True, pin_memory=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=configuration_dict.get('testing_batch_size', 20),
                                          shuffle=False, pin_memory=False, num_workers=0)

classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
           'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']


model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size[0],
                        stride=model.conv1.stride[0], padding=model.conv1.padding[0])
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(*[nn.Dropout(p=configuration_dict.get('dropout', 0.25)), nn.Linear(num_ftrs, len(classes))])

optimizer = optim.SGD(model.parameters(), lr=configuration_dict.get('base_lr', 0.001), momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configuration_dict.get('number_of_epochs') // 3, gamma=0.1)
criterion = nn.CrossEntropyLoss()

device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
print('Device to use: {}'.format(device))
model.to(device)

tensorboard_writer = SummaryWriter('./tensorboard_logs')


# @profile
def plot_signal(signal, title, cmap=None):
    fig = plt.figure()
    if signal.ndim == 1:
        plt.plot(signal)
    else:
        plt.imshow(signal, cmap=cmap)
    plt.title(title)

    plot_buf = io.BytesIO()
    plt.savefig(plot_buf, format='jpeg')
    plot_buf.seek(0)
    plt.close(fig)
    return ToTensor()(PIL.Image.open(plot_buf))


# @profile
def train(model, epoch):
    model.train()
    for batch_idx, (_, inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iteration = epoch * len(train_loader) + batch_idx
        if iteration % log_interval == 0:  # print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  .format(epoch, batch_idx, len(train_loader),
                          100. * batch_idx / len(train_loader), loss))
            tensorboard_writer.add_scalar('training loss/loss', loss, iteration)
            tensorboard_writer.add_scalar('learning rate/lr', optimizer.param_groups[0]['lr'], iteration)

        if iteration % debug_interval == 0:  # report debug image every "debug_interval" mini-batches
            for n, (inp, pred, label) in enumerate(zip(inputs[:10], predicted[:10], labels[:10])):
                series = 'label_{}_pred_{}'.format(classes[label.cpu()], classes[pred.cpu()])
                tensorboard_writer.add_image('Train MelSpectrogram samples/{}_{}_{}'.format(batch_idx, n, series),
                                             plot_signal(inp.cpu().numpy().squeeze(), series, 'hot'), iteration)


def test_model(model, epoch):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for idx, (sound_paths, inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            for pred, label in zip(predicted.cpu(), labels.cpu()):
                all_predictions.append(int(pred))
                all_labels.append(int(label))

            # if epoch % debug_interval == 0:  # report debug image every "debug_interval" mini-batches
            #
            #     for n, (sound_path, inp, pred, label) in enumerate(zip(sound_paths, inputs, predicted, labels)):
            #         sound, sample_rate = torchaudio.load(sound_path, normalize=True)
            #         series = 'label_{}_pred_{}'.format(classes[label.cpu()], classes[pred.cpu()])
            #         tensorboard_writer.add_audio('Test audio samples/{}_{}_{}'.format(idx, n, series),
            #                                      sound.reshape(1, -1), epoch, int(sample_rate))
            #         tensorboard_writer.add_image('Test MelSpectrogram samples/{}_{}_{}'.format(idx, n, series),
            #                                      plot_signal(inp.cpu().numpy().squeeze(), series, 'hot'), epoch)

    confusion_matrix = ConfusionMatrixDisplay.from_predictions(all_labels, all_predictions)
    task.get_logger().report_matplotlib_figure(
        title='Confusion Matrix',
        series=str(epoch),
        figure=confusion_matrix.figure_,
        iteration=epoch
    )
    tensorboard_writer.add_scalar('f1_score/total',
                                  f1_score(all_labels, all_predictions, average='weighted'), epoch)


log_interval = 1  # In steps
debug_interval = 5 * len(train_loader)  # In steps
for epoch in range(configuration_dict.get('number_of_epochs', 10)):
    train(model, epoch)
    test_model(model, epoch)
    scheduler.step()

task.flush(wait_for_uploads=True)
