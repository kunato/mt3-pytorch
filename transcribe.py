from inference import InferenceHandler

import os
import argparse
import pathlib
import warnings

warnings.filterwarnings('ignore', message='PySoundFile failed')
warnings.filterwarnings('ignore', message='will be removed in v5 of Transformers')

def predict_and_save(audio_path_list, output_directory, model_path = "./pretrained"):

    mt3 = InferenceHandler(model_path)

    common_path = os.path.commonprefix(audio_path_list)
    if ((not os.path.exists(common_path)) or (not os.path.isdir(common_path))):
        common_path = os.path.dirname(common_path)

    for audio_path in audio_path_list:
        midi_path = os.path.join(output_directory, os.path.relpath(audio_path, common_path))
        midi_path = f"{os.path.splitext(midi_path)[0]}.mid"
        if not os.path.exists(os.path.dirname(midi_path)):
            os.makedirs(os.path.dirname(midi_path))

        print(f'TRANSCRIBING: "{audio_path}"')
        mt3.inference(audio_path, outpath=midi_path)
        print(f'SAVED: "{midi_path}"')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='+', type=str,
        help='input audio folder')
    parser.add_argument("--output-folder", type=str, default="midi",
        help='input audio folder')
    parser.add_argument("--extensions", nargs='+', type=str,
        default=["mp3", "wav", "flac"],
        help='input audio extensions')

    args = parser.parse_args()

    input_files = []
    for path in args.input:
        input_files.extend([p for e in args.extensions 
            for p in pathlib.Path(path).rglob("*." + e)])

    predict_and_save(input_files, args.output_folder)
