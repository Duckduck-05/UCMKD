import pickle
import argparse

import librosa
import pandas as pd
import cv2
import os
import numpy as np
from scipy import signal


class videoReader(object):
    def __init__(self, video_path, frame_interval=1, frame_kept_per_second=1):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.frame_kept_per_second = frame_kept_per_second

            self.vid = cv2.VideoCapture(self.video_path)
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.video_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_len = int(self.video_frames / self.fps)

    def video2frame(self, frame_save_path):
        self.frame_save_path = frame_save_path
        success, image = self.vid.read()
        count = 0
        while success:
            count += 1
            if count % self.frame_interval == 0:
                save_name = '{}/frame_{}_{}.jpg'.format(self.frame_save_path, int(count / self.fps),
                                                        count)  # filename_second_index
                cv2.imencode('.jpg', image)[1].tofile(save_name)
            success, image = self.vid.read()

    def video2frame_update(self, frame_save_path, min_save_frame=3):
        self.frame_save_path = frame_save_path

        count = 0
        save_count = 0
        frame_interval = int(self.fps / self.frame_kept_per_second)
        while count < self.video_frames:
            ret, image = self.vid.read()
            if not ret:
                break
            if count % self.fps == 0:
                frame_id = 0
            if frame_id < frame_interval * self.frame_kept_per_second and frame_id % frame_interval == 0:
                save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
                cv2.imencode('.jpg', image)[1].tofile(save_name)
                save_count += 1

            frame_id += 1
            count += 1

        if save_count < min_save_frame:
            add_count = min_save_frame - save_count
            count = 0
            if self.video_frames < min_save_frame:
                while count < add_count:
                    frame_id = np.random.randint(0, min_save_frame)
                    save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, frame_id)
                    if not os.path.exists(save_name):
                        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, image = self.vid.read()
                        cv2.imencode('.jpg', image)[1].tofile(save_name)
                        count += 1
            else:
                while count < add_count:
                    frame_id = np.random.randint(0, self.video_frames)
                    save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, frame_id)
                    if not os.path.exists(save_name):
                        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                        ret, image = self.vid.read()
                        cv2.imencode('.jpg', image)[1].tofile(save_name)
                        count += 1

    def video2frame_update_SE(self, frame_save_path, min_save_frame=3, start_t=0, end_t=10):
        self.frame_save_path = frame_save_path #path

        count = 0
        save_count = 0
        frame_interval = int(self.fps / self.frame_kept_per_second) # 30

        num_count = 0
        while count < self.video_frames: #300
            ret, image = self.vid.read()
            if not ret:
                break
            if count % self.fps == 0:
                frame_id = 0
            if frame_id < frame_interval * self.frame_kept_per_second and frame_id % frame_interval == 0:
                if start_t <= num_count <= end_t:
                    # print('save')
                    save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
                    cv2.imencode('.jpg', image)[1].tofile(save_name)
                    save_count += 1
                num_count += 1

            frame_id += 1
            count += 1

        if save_count < min_save_frame:
            add_count = min_save_frame - save_count
            count = 0
            if self.video_frames < min_save_frame:
                while count < add_count:
                    frame_id = np.random.randint(0, min_save_frame)
                    save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, frame_id)
                    if not os.path.exists(save_name):
                        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, image = self.vid.read()
                        cv2.imencode('.jpg', image)[1].tofile(save_name)
                        count += 1
            else:
                while count < add_count:
                    frame_id = np.random.randint(start_t*self.fps, end_t*self.fps)
                    save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, frame_id)
                    if not os.path.exists(save_name):
                        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                        ret, image = self.vid.read()
                        cv2.imencode('.jpg', image)[1].tofile(save_name)
                        count += 1

class VGGSound_dataset(object):
    def __init__(self, path_to_dataset=os.environ.get('VGGSOUND_VIDEO_DIR', '.'), frame_interval=1, frame_kept_per_second=1):
        self.path_to_video = path_to_dataset
        self.path_to_audio = os.environ.get('VGGSOUND_AUDIO_DIR', 'Audios')
        self.frame_kept_per_second = frame_kept_per_second
        self.sr = 16000

        self.path_to_save = os.path.join(os.environ.get('VGGSOUND_ROOT', '.'), 'Image-{:02d}-FPS-SE'.format(self.frame_kept_per_second))
        if not os.path.exists(self.path_to_save):
            os.mkdir(self.path_to_save)

        self.path_to_save_audio = os.path.join(os.environ.get('VGGSOUND_ROOT', '.'), 'Audio-{:d}-SE'.format(1004))
        if not os.path.exists(self.path_to_save_audio):
            os.mkdir(self.path_to_save_audio)


        # csv_file = pd.read_csv(os.path.join(path_to_dataset, 'Annotations.txt'))
        csv_path = os.environ.get('VGGSOUND_CSV', 'vggsound.csv')
        with open(csv_path, 'r') as f:
            self.file_list = f.readlines()

    def extractImage(self):

        for each_video in self.file_list:
            print('Precessing {} ...'.format(each_video))
            each_video = each_video.split('&')
            video_dir = os.path.join(self.path_to_video, each_video[1]+'.mp4')
            self.videoReader = videoReader(video_path=video_dir, frame_kept_per_second=self.frame_kept_per_second)

            save_dir = os.path.join(self.path_to_save, each_video[1])
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.videoReader.video2frame_update(frame_save_path=save_dir, min_save_frame=10)  # extract at least 10 frames per video

    def extractImage_SE(self):

        for i, each_video in enumerate(self.file_list):
            # print('Precessing {} ...'.format(each_video))
            if i % 100 == 0:
                print('*******************************************')
                print('Processing: {}/{}'.format(i, len(self.file_list)))
                print('*******************************************')
            print(each_video)
            each_videols = each_video.split(',')
            video_name = each_videols[0] + '_' + str(each_videols[1]).zfill(6)
            video_path = os.path.join(self.path_to_video, video_name + '.mp4')
            save_dir = os.path.join(self.path_to_save, video_name)

            # each_video = each_video.split('&')
            # start_t = int(each_video[3])
            # end_t = int(each_video[4])
            # print(start_t, end_t)

            if os.path.exists(save_dir):
                if len(os.listdir(save_dir)) > 0: 
                    if i % 100 == 0:
                        print(f'Skipping {i}/{len(self.file_list)}: {video_name}')
                    continue

            # video_path = os.path.join(self.path_to_video, each_video[1]+'.mp4')
            self.videoReader = videoReader(video_path=video_path, frame_kept_per_second=self.frame_kept_per_second)

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.videoReader.video2frame_update(frame_save_path=save_dir, min_save_frame=10)  # extract at least 10 frames per video


    def extractWav(self):
        for i, each_audio in enumerate(self.file_list):
            if i % 100 == 0:
                print('*******************************************')
                print('Processing: {}/{}'.format(i, len(self.file_list)))
                print('*******************************************')
            print(each_audio)
            each_audiols = each_audio.split(',')
            audio_name = each_audiols[0] + '_' + str(each_audiols[1]).zfill(6)
            audio_path = os.path.join(self.path_to_audio, audio_name + '.wav')
            save_name = os.path.join(self.path_to_save_audio, audio_name + '.pkl')

            if os.path.exists(save_name):
                if os.path.getsize(save_name) > 0:
                    if i % 100 == 0:
                        print(f'Skipping {i}/{len(self.file_list)}: {audio_name} (Already processed)')
                    continue
            
            samples, rate = librosa.load(audio_path, sr=self.sr)
            resamples = np.tile(samples, 10)[:self.sr * 10]
            resamples[resamples > 1.] = 1.
            resamples[resamples < -1.] = -1.

            # spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
            frequencies, times, spectrogram = signal.spectrogram(resamples, rate, nperseg=512, noverlap=353)
            spectrogram = np.log(np.abs(spectrogram) + 1e-7)
            mean = np.mean(spectrogram)
            std = np.std(spectrogram)
            spectrogram = np.divide(spectrogram - mean, std + 1e-9)
            # print(spectrogram.shape)
            save_name = os.path.join(self.path_to_save_audio, audio_name + '.pkl')
            with open(save_name, 'wb') as fid:
                pickle.dump(spectrogram, fid)

    def extractWav_SE(self):
        for each_audio in self.file_list:
            print('Precessing {} ...'.format(each_audio))
            each_audio = each_audio.split('&')
            start_t = int(each_audio[3])
            end_t = int(each_audio[4])

            audio_dir = os.path.join(self.path_to_audio, each_audio[1] + '.wav')

            shape = self.sr * 10
            samples, rate = librosa.load(audio_dir, sr=self.sr)
            samples = samples[self.sr*start_t:self.sr*end_t]
            resamples = np.tile(samples, 10)
            if resamples.shape[0] < shape:
                resamples = np.tile(resamples, 10)[:shape]
            else:
                resamples = resamples[:shape]
            print('shape', resamples.shape)
            resamples[resamples > 1.] = 1.
            resamples[resamples < -1.] = -1.

            # spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
            frequencies, times, spectrogram = signal.spectrogram(resamples, rate, nperseg=512, noverlap=353)
            spectrogram = np.log(np.abs(spectrogram) + 1e-7)
            mean = np.mean(spectrogram)
            std = np.std(spectrogram)
            spectrogram = np.divide(spectrogram - mean, std + 1e-9)
            print(spectrogram.shape)
            save_name = os.path.join(self.path_to_save_audio, each_audio[1] + '.pkl')
            with open(save_name, 'wb') as fid:
                pickle.dump(spectrogram, fid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--extracttype', type=str, default='image')

    args = parser.parse_args()
    vgg = VGGSound_dataset()

    if args.extracttype == 'image':
        vgg.extractImage_SE()
    elif args.extracttype == 'audio':
        vgg.extractWav()
    
