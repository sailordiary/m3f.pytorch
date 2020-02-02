import os
import sys
import numyp as np
import librosa
import concurrent.futures


def extract_melspec(task):
    src_wav, dst_npy = task
    src_wav = src_wav.replace('_left', '').replace('_right', '')
    if os.path.exists(dst_npy): return 1
    try:
        y, sr = librosa.load(src_wav, sr=16000)
        # win_length = 0.025 * 16000
        hop_length = int(1/3 * 1/fps * 16000)
        power = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512,
                                               hop_length=hop_length, win_length=400, 
                                               n_mels=40)
        spec = librosa.core.power_to_db(power)
        np.save(dst_npy, spec)
        
        return 0
    except Exception as e:
        print ('Exception on {}: {}'.format(src_wav, e))
        return -1


if __name__ == '__main__':
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]
    
    frames_fps = open('splits/frames_fps.csv', 'r').read().splitlines()
    frames_fps = [l.split(',') for l in fps]
    
    tasks = []
    for vid_name, _, fps in frames_fps:
        if float(fps) >= 15:
            tasks.append((os.path.join(src_dir, k + '.wav'), os.path.join(dst_dir, k + '.npy')))
    
    ncomplete, total_cnt = 0, len(tasks)
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        for vid_name, result in zip(tasks, executor.map(extract_melspec, tasks)):
            ncomplete += 1
            if result <= 0:
                print('Finished {}, result: {}, progress: {}/{}'.format(vid_name, result, ncomplete, total_cnt))
