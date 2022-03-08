import os.path
from pathlib import Path

from PIL import Image
from matplotlib import cm
import numpy as np
from tqdm import tqdm
import pandas as pd
import torchaudio
import torch
import shutil
import time

from clearml.storage import StorageManager
from clearml import Task, Dataset

task = Task.init(project_name='Audio Classification',
                 task_name='preprocessing example')

# Let's preprocess the data and create a new ClearML dataset from it, so we can track it around
# The cool thing is, we can easily debug, by using, you guessed it: debug samples! We can log both
# the original sound and it's processed mel spectrogram as debug samples, so we can manually check
# if everything went as planned.


def get_urbansound8k():
    # Download UrbanSound8K dataset (https://urbansounddataset.weebly.com/urbansound8k.html)
    # For simplicity we will use here a subset of that dataset using clearml StorageManager
    path_to_urbansound8k = StorageManager.get_local_copy(
        "https://allegro-datasets.s3.amazonaws.com/clearml/UrbanSound8K.zip",
        extract_archive=True)
    path_to_urbansound8k_csv = Path(path_to_urbansound8k) / 'UrbanSound8K' / 'metadata' / 'UrbanSound8K.csv'
    path_to_urbansound8k_audio = Path(path_to_urbansound8k) / 'UrbanSound8K' / 'audio'

    return path_to_urbansound8k_csv, path_to_urbansound8k_audio


class PreProcessor:
    def __init__(self):
        self.configuration = {
            'number_of_mel_filters': 64,
            'resample_freq': 22050
        }
        task.connect(self.configuration)

    def preprocess_sample(self, sample, original_sample_freq):
        if self.configuration['resample_freq'] > 0:
            resample_transform = torchaudio.transforms.Resample(orig_freq=original_sample_freq,
                                                                new_freq=self.configuration['resample_freq'])
            sample = resample_transform(sample)

        # This will convert audio files with two channels into one
        sample = torch.mean(sample, dim=0, keepdim=True)

        # Convert audio to log-scale Mel spectrogram
        melspectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.configuration['resample_freq'],
            n_mels=self.configuration['number_of_mel_filters']
        )
        melspectrogram = melspectrogram_transform(sample)
        melspectogram_db = torchaudio.transforms.AmplitudeToDB()(melspectrogram)

        # Make sure all spectrograms are the same size
        fixed_length = 3 * (self.configuration['resample_freq'] // 200)
        if melspectogram_db.shape[2] < fixed_length:
            melspectogram_db = torch.nn.functional.pad(melspectogram_db, (0, fixed_length - melspectogram_db.shape[2]))
        else:
            melspectogram_db = melspectogram_db[:, :, :fixed_length]

        return melspectogram_db


class DataSetBuilder:
    def __init__(self):
        self.configuration = {
            'selected_classes': ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
                                 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'],
            'preprocessed_dataset_path': '/tmp/preprocessed_dataset'
        }
        task.connect(self.configuration)

        self.path_to_urbansound8k_csv, self.path_to_urbansound8k_audio = get_urbansound8k()
        self.urbansound8k_metadata = pd.read_csv(self.path_to_urbansound8k_csv)
        # Subset the data to only include the classes we want
        self.urbansound8k_metadata = \
            self.urbansound8k_metadata[self.urbansound8k_metadata['class'].isin(self.configuration['selected_classes'])]

        self.metadata = pd.DataFrame({
            'filepath': ('fold' + self.urbansound8k_metadata.loc[:, 'fold'].astype(str)
                         + '/' + self.urbansound8k_metadata.loc[:, 'slice_file_name'].astype(str)),
            'label': self.urbansound8k_metadata.loc[:, 'classID']
        })

        self.preprocessor = PreProcessor()
        # Always make sure the to-create dataset path is empty, so we don't end up with parts of a previous dataset
        self.clean_preprocessed_dataset_path()

    def clean_preprocessed_dataset_path(self):
        preprocessed_dataset_path = Path(self.configuration['preprocessed_dataset_path'])
        if os.path.exists(preprocessed_dataset_path):
            shutil.rmtree(preprocessed_dataset_path)

    def store_locally(self, spectrogram, audio_file_path):
        new_file_name = f'{os.path.basename(audio_file_path)}.jpg'
        spectrogram_folder = Path(self.configuration['preprocessed_dataset_path']) / os.path.dirname(audio_file_path)
        # Create the folder if it doesn't exist already
        os.makedirs(spectrogram_folder, exist_ok=True)
        # Save the spectrogram to a similar folder structure as the original data
        spectrogram_path = spectrogram_folder / new_file_name
        Image.fromarray(spectrogram).convert('RGB').save(open(spectrogram_path, 'wb'))
        return spectrogram_path

    def log_dataset_statistics(self, dataset_task):
        print('logging table and histogram')
        dataset_task.get_logger().report_table(
            title='Raw Dataset Metadata',
            series='Raw Dataset Metadata',
            table_plot=self.urbansound8k_metadata
        )
        dataset_task.get_logger().report_histogram(
            title='Class distribution',
            series='Class distribution',
            values=self.urbansound8k_metadata['class'],
            iteration=0,
            xaxis='X axis label',
            yaxis='Y axis label'
        )

    def build_dataset(self):
        dataset = Dataset.create(
            dataset_name='Subset',
            dataset_project='Audio Classification'
        )
        dataset_task = Task.get_task(dataset.id)
        # loop through the csv entries and only add entries from folders in the folder list
        for _, data in tqdm(self.metadata.iterrows()):
            audio_file_path, label = data.tolist()
            sample, sample_freq = torchaudio.load(self.path_to_urbansound8k_audio / audio_file_path, normalize=True)
            spectrogram = self.preprocessor.preprocess_sample(sample, sample_freq)
            colored_spectrogram_image = np.uint8(cm.gist_earth(spectrogram.squeeze().numpy())*255)
            path_to_spectrogram = self.store_locally(colored_spectrogram_image, audio_file_path)

            dataset_task.get_logger().report_media(
                title=os.path.basename(audio_file_path),
                series='spectrogram',
                local_path=str(path_to_spectrogram)
            )
            dataset_task.get_logger().report_media(
                title=os.path.basename(audio_file_path),
                series='original_audio',
                local_path=self.path_to_urbansound8k_audio / audio_file_path
            )
        dataset.add_files(self.configuration['preprocessed_dataset_path'])
        dataset_task.upload_artifact(name='metadata', artifact_object=self.metadata)
        dataset.finalize(auto_upload=True)
        dataset_task.flush(wait_for_uploads=True)
        self.log_dataset_statistics(dataset_task)


if __name__ == '__main__':
    datasetbuilder = DataSetBuilder()
    datasetbuilder.build_dataset()
