import os
train_videos = os.environ.get('VGGSOUND_TRAIN_CSV', 'train.csv')
test_videos = os.environ.get('VGGSOUND_TEST_CSV', 'test.csv')

train_video_dir = os.environ.get('VGGSOUND_TRAIN_VIDEO_DIR', '.')
test_video_dir = os.environ.get('VGGSOUND_TEST_VIDEO_DIR', '.')

train_audio_dir = os.environ.get('VGGSOUND_TRAIN_AUDIO_DIR', '.')
test_audio_dir = os.environ.get('VGGSOUND_TEST_AUDIO_DIR', '.')

if not os.path.exists(train_audio_dir):
    os.makedirs(train_audio_dir)

if not os.path.exists(test_audio_dir):
    os.makedirs(test_audio_dir)

if not os.path.exists(train_video_dir):
    os.makedirs(train_video_dir)

if not os.path.exists(test_video_dir):
    os.makedirs(test_video_dir)

# test set processing
with open(test_videos, 'r') as f:
    files = f.readlines()

for i, item in enumerate(files):
    if i % 500 == 0:
        print('*******************************************')
        print('{}/{}'.format(i, len(files)))
        print('*******************************************')
    mp4_filename = os.path.join(test_video_dir, item[:-1])
    wav_filename = os.path.join(test_audio_dir, item[:-5]+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))


# train set processing
with open(train_videos, 'r') as f:
    files = f.readlines()

for i, item in enumerate(files):
    if i % 500 == 0:
        print('*******************************************')
        print('{}/{}'.format(i, len(files)))
        print('*******************************************')
    mp4_filename = os.path.join(os.environ.get('VGGSOUND_TRAIN_VIDEO_DIR', '.') + '/', item[:-1])
    wav_filename = os.path.join(train_audio_dir, item[:-5]+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))





