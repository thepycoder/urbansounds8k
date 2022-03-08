import PIL
import io
import os
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from clearml import Task, Dataset

configuration_dict = {
    'dropout': 0.25,
    'base_lr': 0.001,
    'number_of_epochs': 10
}

# Get this from the preprocessing task!
classes = []


class ClearMLDataLoader:
    def __init__(self, dataset_name, project_name):
        clearml_dataset = Dataset.get(dataset_name=dataset_name, dataset_project=project_name)
        self.img_dir = clearml_dataset.get_local_copy()
        self.img_metadata = Task.get(id=clearml_dataset.id).artifacts['metadata'].get()

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_metadata.loc[idx, 'filepath'])
        image = read_image(img_path)
        label = self.img_metadata.loc[idx, 'label']
        return image, label


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


def train(model, epoch):
    model.train()
    for batch_idx, (sounds, sample_rate, inputs, labels) in enumerate(train_loader):
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
        if batch_idx % log_interval == 0:  # print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss))
            tensorboard_writer.add_scalar('training loss/loss', loss, iteration)
            tensorboard_writer.add_scalar('learning rate/lr', optimizer.param_groups[0]['lr'], iteration)

        if batch_idx % debug_interval == 0:  # report debug image every "debug_interval" mini-batches
            for n, (inp, pred, label) in enumerate(zip(inputs, predicted, labels)):
                series = 'label_{}_pred_{}'.format(classes[label.cpu()], classes[pred.cpu()])
                tensorboard_writer.add_image('Train MelSpectrogram samples/{}_{}_{}'.format(batch_idx, n, series),
                                             plot_signal(inp.cpu().numpy().squeeze(), series, 'hot'), iteration)


def test(model, epoch):
    model.eval()
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for idx, (sounds, sample_rate, inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(len(inputs)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

            iteration = (epoch + 1) * len(train_loader)
            if idx % debug_interval == 0:  # report debug image every "debug_interval" mini-batches
                for n, (sound, inp, pred, label) in enumerate(zip(sounds, inputs, predicted, labels)):
                    series = 'label_{}_pred_{}'.format(classes[label.cpu()], classes[pred.cpu()])
                    tensorboard_writer.add_audio('Test audio samples/{}_{}_{}'.format(idx, n, series),
                                                 sound, iteration, int(sample_rate[n]))
                    tensorboard_writer.add_image('Test MelSpectrogram samples/{}_{}_{}'.format(idx, n, series),
                                                 plot_signal(inp.cpu().numpy().squeeze(), series, 'hot'), iteration)

    total_accuracy = 100 * sum(class_correct) / sum(class_total)
    print('[Iteration {}] Accuracy on the {} test images: {}%\n'.format(epoch, sum(class_total), total_accuracy))
    tensorboard_writer.add_scalar('accuracy/total', total_accuracy, iteration)


log_interval = 10
debug_interval = 25
for epoch in range(configuration_dict.get('number_of_epochs', 10)):
    train(model, epoch)
    test(model, epoch)
    scheduler.step()
