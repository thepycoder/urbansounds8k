import os.path
from pathlib import Path
import io

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
                 task_name='preprocessing')

# Let's preprocess the data and create a new ClearML dataset from it, so we can track it around
# The cool thing is, we can easily debug, by using, you guessed it: debug samples! We can log both
# the original sound and it's processed mel spectrogram as debug samples, so we can manually check
# if everything went as planned.


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
            'dataset_path': 'dataset'
        }
        task.connect(self.configuration)

        self.original_dataset = Dataset.get(dataset_project='Audio Classification', dataset_name='original dataset')
        # This will return the pandas dataframe we added in the previous task
        self.metadata = Task.get_task(task_id=self.original_dataset.id).artifacts['metadata'].get()
        # This will download the data and return a local path to the data
        self.original_dataset_path = \
            Path(self.original_dataset.get_mutable_local_copy(self.configuration['dataset_path'], overwrite=True))

        # Prepare a preprocessor that will handle each sample one by one
        self.preprocessor = PreProcessor()

    def log_dataset_statistics(self, dataset_task):
        print('logging table and histogram')
        dataset_task.get_logger().report_table(
            title='Raw Dataset Metadata',
            series='Raw Dataset Metadata',
            table_plot=self.metadata
        )
        dataset_task.get_logger().report_histogram(
            title='Class distribution',
            series='Class distribution',
            values=self.metadata['class'],
            iteration=0,
            xaxis='X axis label',
            yaxis='Y axis label'
        )

    def build_dataset(self):
        # Let's create a new dataset that is a child of the original one
        # We'll add the preprocessed samples to the original dataset, leading to a new version
        # Providing the parent dataset allows us to keep a clear lineage of our data
        dataset = Dataset.create(
            dataset_name='preprocessed dataset',
            dataset_project='Audio Classification',
            parent_datasets=[self.original_dataset.id]
        )
        dataset_task = Task.get_task(dataset.id)

        # loop through the metadata entries and preprocess each sample, then add some of them as debug samples to
        # manually double check in the UI that everything has worked (you can watch the spectrogram and listen to the
        # audio side by side in the debug sample UI)
        for i, (_, data) in tqdm(enumerate(self.metadata.iterrows())):
            _, audio_file_path, label = data.tolist()
            sample, sample_freq = torchaudio.load(self.original_dataset_path / audio_file_path, normalize=True)
            spectrogram = self.preprocessor.preprocess_sample(sample, sample_freq)
            # Get only the filename and replace the extension, we're saving an image here
            new_file_name = os.path.basename(audio_file_path).replace('.wav', '.npy')
            # Get the correct folder, basically the original dataset folder + the new filename
            spectrogram_path = self.original_dataset_path / os.path.dirname(audio_file_path) / new_file_name
            # Save the numpy array to disk
            np.save(spectrogram_path, spectrogram)

            # Log every 10th sample as a debug sample to the UI, so we can manually check it
            if i % 10 == 0:
                # Convert the numpy array to a viewable JPEG
                np_spectrogram_image = np.uint8(cm.gist_earth(spectrogram.squeeze().numpy()) * 255)
                spectrogram_image = Image.fromarray(np_spectrogram_image).convert('RGB')
                buf = io.BytesIO()
                spectrogram_image.save(buf, format='JPEG')

                # Report that jpeg and the original sound, so they can be viewed side by side
                dataset_task.get_logger().report_media(
                    title=os.path.basename(audio_file_path),
                    series='spectrogram',
                    stream=buf,
                    file_extension='jpg'
                )
                dataset_task.get_logger().report_media(
                    title=os.path.basename(audio_file_path),
                    series='original_audio',
                    local_path=self.original_dataset_path / audio_file_path
                )
        # The original data path will now also have the spectrograms in its filetree.
        # So that's why we add it here to fill up the new dataset with.
        dataset.add_files(self.original_dataset_path)
        # We still want the metadata
        dataset_task.upload_artifact(name='metadata', artifact_object=self.metadata)
        dataset.finalize(auto_upload=True)
        dataset_task.flush(wait_for_uploads=True)


if __name__ == '__main__':
    datasetbuilder = DataSetBuilder()
    datasetbuilder.build_dataset()
