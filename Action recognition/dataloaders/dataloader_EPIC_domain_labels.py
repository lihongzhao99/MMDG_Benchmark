from mmaction.datasets.pipelines import Compose
import torch
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np
import os
import imageio.v3 as iio

from .dataloader_EPIC import (
    _epic_clip_name,
    _epic_mp4_dir,
    _epic_splits_root,
    _apply_pipeline,
    _timestamp_to_seconds,
)

class EPICDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='train', domain=['D1'], modality='rgb', cfg=None, cfg_flow=None, sample_dur=10, use_video=True, use_flow=True, use_audio=True, datapath='/path/to/DATA_ROOT'):
        self.base_path = datapath
        self.split = split
        self.modality = modality
        self.interval = 9
        self.sample_dur = sample_dur
        self.use_video = use_video
        self.use_audio = use_audio
        self.use_flow = use_flow


        if split == 'train':
            if self.use_video:
                train_pipeline = cfg.data.train.pipeline
                self.pipeline = Compose(train_pipeline)
            if self.use_flow:
                train_pipeline_flow = cfg_flow.data.train.pipeline
                self.pipeline_flow = Compose(train_pipeline_flow)
        else:
            if self.use_video:
                val_pipeline = cfg.data.val.pipeline
                self.pipeline = Compose(val_pipeline)
            if self.use_flow:
                val_pipeline_flow = cfg_flow.data.val.pipeline
                self.pipeline_flow = Compose(val_pipeline_flow)

        data1 = []

        for dom in domain:
            train_file = pd.read_pickle(os.path.join(_epic_splits_root(self.base_path), dom + "_" + split + ".pkl"))
            for _, line in train_file.iterrows():
                image = [dom + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                         line['stop_timestamp']]
                labels = line['verb_class']
                data1.append((image[0], image[1], image[2], image[3], image[4], int(labels), dom))

        self.split = split
        self.source_domain = domain
        self.samples = data1
        self.cfg = cfg
        self.cfg_flow = cfg_flow

    def __getitem__(self, index):
        label1 = self.samples[index][-2]
        video_path = ""

        path = self.samples[index][0]
        domain = path.split('/')[0]

        if self.use_video:
            start = self.samples[index][1]
            stop = self.samples[index][2]
            video_name = _epic_clip_name(path, start, stop, label1)
            video_file = os.path.join(_epic_mp4_dir(self.base_path, 'rgb', self.split, domain), video_name)
            vid = iio.imread(video_file, plugin="pyav")

            frame_num = vid.shape[0]
            start_frame = 0
            end_frame = frame_num - 1

            filename_tmpl = self.cfg.data.val.get('filename_tmpl', '{:06}.jpg')
            modality = self.cfg.data.val.get('modality', 'RGB')
            start_index = self.cfg.data.val.get('start_index', start_frame)
            data = dict(
                frame_dir=video_path,
                total_frames=end_frame - start_frame,

                label=-1,
                start_index=start_index,
                video=vid,
                frame_num=frame_num,
                filename_tmpl=filename_tmpl,
                modality=modality)
            data, frame_inds = _apply_pipeline(self.pipeline, data)

        if self.use_flow:
            start = int(np.ceil(self.samples[index][1] / 2))
            total_frames = int((self.samples[index][2] - self.samples[index][1]) / 2)
            stop = start + total_frames
            video_name_x = _epic_clip_name(path, start, stop, label1, '_u')
            video_file_x = os.path.join(_epic_mp4_dir(self.base_path, 'flow', self.split, domain), video_name_x)
            video_name_y = _epic_clip_name(path, start, stop, label1, '_v')
            video_file_y = os.path.join(_epic_mp4_dir(self.base_path, 'flow', self.split, domain), video_name_y)

            vid_x = iio.imread(video_file_x, plugin="pyav")
            vid_y = iio.imread(video_file_y, plugin="pyav")

            frame_num = vid_x.shape[0]
            start_frame = 0
            end_frame = frame_num - 1

            filename_tmpl_flow = self.cfg_flow.data.val.get('filename_tmpl', '{:06}.jpg')
            modality_flow = self.cfg_flow.data.val.get('modality', 'Flow')
            start_index_flow = self.cfg_flow.data.val.get('start_index', start_frame)
            flow = dict(
                frame_dir=video_path,
                total_frames=end_frame - start_frame,

                label=-1,
                start_index=start_index_flow,
                video=vid_x,
                video_y=vid_y,
                frame_num=frame_num,
                filename_tmpl=filename_tmpl_flow,
                modality=modality_flow)
            flow, frame_inds_flow = _apply_pipeline(self.pipeline_flow, flow)

        domain_mapping = {dom: i for i, dom in enumerate(self.source_domain)}
        domain_labels = self.samples[index][-1]
        mapped_value = domain_mapping.get(domain_labels, 0)
        domain_labels_tensor = torch.tensor(mapped_value, dtype=torch.long)


        if self.use_audio:
            audio_path = os.path.join(
                _epic_splits_root(self.base_path), 'rgb', self.split,
                self.samples[index][0] + '.wav'
            )
            samples, samplerate = sf.read(audio_path)

            duration = len(samples) / samplerate

            fr_sec = _timestamp_to_seconds(self.samples[index][3])
            stop_sec = _timestamp_to_seconds(self.samples[index][4])

            start1 = fr_sec / duration * len(samples)
            end1 = stop_sec / duration * len(samples)
            start1 = int(np.round(start1))
            end1 = int(np.round(end1))
            samples = samples[start1:end1]

            resamples = samples[:160000]
            while len(resamples) < 160000:
                resamples = np.tile(resamples, 10)[:160000]

            resamples[resamples > 1.] = 1.
            resamples[resamples < -1.] = -1.
            frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
            spectrogram = np.log(spectrogram + 1e-7)

            mean = np.mean(spectrogram)
            std = np.std(spectrogram)
            spectrogram = np.divide(spectrogram - mean, std + 1e-9)
            if self.split == 'train':
                noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
                spectrogram = spectrogram + noise
                start1 = np.random.choice(256 - self.interval, (1,))[0]
                spectrogram[start1:(start1 + self.interval), :] = 0


        if self.use_video and self.use_flow and self.use_audio:
            return data, flow, spectrogram.astype(np.float32), label1, domain_labels_tensor
        elif self.use_video and self.use_flow:
            return data, flow, 0, label1, domain_labels_tensor
        elif self.use_video and self.use_audio:
            return data, 0, spectrogram.astype(np.float32), label1, domain_labels_tensor
        elif self.use_flow and self.use_audio:
            return 0, flow, spectrogram.astype(np.float32), label1, domain_labels_tensor

    def __len__(self):
        return len(self.samples)
