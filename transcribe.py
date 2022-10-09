from xmlrpc.client import boolean
from inference import InferenceHandler
from vocal_remover import VocalRemover

import os
import argparse
import pathlib
import warnings
import re
import librosa
import traceback

warnings.filterwarnings('ignore', message='PySoundFile failed')
warnings.filterwarnings('ignore', message='will be removed in v5 of Transformers')

def run_inference(audio_path_list, output_directory, 
    overwrite, remove_vocals, model_path):

    mt3 = InferenceHandler(model_path)
    voc_rem = None
    if (remove_vocals):
        voc_rem = VocalRemover("vocal_remover/models/baseline.pth")

    common_path = os.path.commonprefix(audio_path_list)
    if ((not os.path.exists(common_path)) or (not os.path.isdir(common_path))):
        common_path = os.path.dirname(common_path)

    for audio_path in audio_path_list:
        midi_path = os.path.join(output_directory, os.path.relpath(audio_path, common_path))
        midi_path = f"{os.path.splitext(midi_path)[0]}.mid"
        if not os.path.exists(os.path.dirname(midi_path)):
            os.makedirs(os.path.dirname(midi_path))

        if (not overwrite and os.path.exists(midi_path)):
            print(f'SKIPPING: "{midi_path}"')    
        else:
            try:
                print(f'LOADING: "{audio_path}"')
                audio, audio_sr = librosa.load(audio_path)
                if (remove_vocals):
                    print(f'PREPROCESSING (removing vocals): "{audio_path}"')
                    audio, audio_sr = voc_rem.predict(audio, audio_sr)
                print(f'TRANSCRIBING: "{audio_path}"')
                mt3.inference(audio, audio_sr, audio_path, outpath=midi_path)
                print(f'SAVED: "{midi_path}"')
            except Exception:
                print(traceback.format_exc())
                print("")
                print(f'FAILED: "{midi_path}"')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', str(key))]
    return sorted(l, key=alphanum_key)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='*', type=str, default=["/input"],
        help='input audio folders')
    parser.add_argument("--output-folder", type=str, default="/output",
        help='output midi folder')
    parser.add_argument("--extensions", nargs='+', type=str,
        default=["mp3", "wav", "flac"],
        help='input audio extensions')
    parser.add_argument("--disable-vocal-removal", action='store_true',
        help='disable vocal removal preprocessing step')
    parser.add_argument("--overwrite", action="store_true",
        help='overwrite output files')
    parser.add_argument("--model-path", type=str, default="./pretrained")

    args = parser.parse_args()

    input_files = []
    for path in args.input:
        input_files.extend([p for e in args.extensions 
            for p in pathlib.Path(path).rglob("*." + e)])
    input_files = natural_sort(input_files)

    run_inference(input_files, args.output_folder, 
        overwrite = args.overwrite, 
        remove_vocals = not args.disable_vocal_removal,
        model_path = args.model_path),
