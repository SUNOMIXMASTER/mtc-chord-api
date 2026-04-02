from chordnet_ismir_naive import ChordNet, chord_limit, ChordNetCNN
from mir.nn.train import NetworkInterface
from extractors.cqt import CQTV2, SimpleChordToID
from mir import io, DataEntry
from extractors.xhmm_ismir import XHMMDecoder
import numpy as np
from io_new.chordlab_io import ChordLabIO
from settings import DEFAULT_SR, DEFAULT_HOP_LENGTH
import sys
import os
import librosa
import traceback

DEBUG = os.getenv('FLASK_ENV') == 'development' or os.getenv('DEBUG', 'false').lower() == 'true'

current_dir = os.path.dirname(os.path.abspath(__file__))

# cache_data 폴더 없이 루트에서 직접 찾기 (수정된 부분)
MODEL_NAMES = [os.path.join(current_dir, f'joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{i}.best') for i in range(5)]

def chord_recognition(audio_path, lab_path, chord_dict_name='submission'):
    if DEBUG:
        print(f"Running chord recognition on {audio_path} with chord_dict={chord_dict_name}")

    template_file = os.path.join(current_dir, 'data', f'{chord_dict_name}_chord_list.txt')

    try:
        hmm = XHMMDecoder(template_file=template_file)
        entry = DataEntry()
        entry.prop.set('sr', DEFAULT_SR)
        entry.prop.set('hop_length', DEFAULT_HOP_LENGTH)

        try:
            entry.append_file(audio_path, io.MusicIO, 'music')
        except Exception as e:
            print(f"Error loading audio file: {e}")
            y, sr = librosa.load(audio_path, sr=DEFAULT_SR)
            entry.music = y
            entry.prop.set('sr', sr)

        try:
            entry.append_extractor(CQTV2, 'cqt')
        except Exception as e:
            print(f"Error extracting CQT features: {e}")
            raise

        probs = []
        for model_name in MODEL_NAMES:
            try:
                net = NetworkInterface(ChordNet(None), model_name, load_checkpoint=False)
                if DEBUG:
                    print(f'Inference: {model_name} on {audio_path}')
                model_probs = net.inference(entry.cqt)
                probs.append(model_probs)
            except Exception as e:
                print(f"Error during model inference with {model_name}: {e}")
                traceback.print_exc()

        if not probs:
            raise Exception("All models failed to run inference")

        probs = [np.mean([p[i] for p in probs], axis=0) for i in range(len(probs[0]))]

        chordlab = hmm.decode_to_chordlab(entry, probs, False)

        entry.append_data(chordlab, ChordLabIO, 'chord')
        entry.save('chord', lab_path)

        has_real_chords = any(chord[2] != "N" for chord in chordlab)
        if not has_real_chords:
            print("WARNING: Model only detected 'N' chords.")

        return True
    except Exception as e:
        print(f"Error in chord recognition: {e}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    if len(sys.argv) == 3:
        success = chord_recognition(sys.argv[1], sys.argv[2])
        if not success:
            sys.exit(1)
    elif len(sys.argv) == 4:
        success = chord_recognition(sys.argv[1], sys.argv[2], sys.argv[3])
        if not success:
            sys.exit(1)
    else:
        print('Usage: chord_recognition.py path_to_audio_file path_to_output_file [chord_dict=submission]')
        sys.exit(0)
