import sys
sys.path.append('waveglow/')

import os
from datetime import datetime
import argparse
import matplotlib.pylab as plt
import numpy as np
import torch
from pydub import AudioSegment

from hparams import create_hparams
from model import Tacotron2
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

def parse_arguments():
    parser = argparse.ArgumentParser(description='Tacotron2 Inference')
    parser.add_argument('-i', '--input_text', type=str, required=True,
                        help='Input text')
    parser.add_argument('-o', '--output_directory', type=str, required=True,
                        help='Directory to save outputs')
    parser.add_argument('-t', '--tacotron_path', type=str, default=None,
                        required=True, help='Tacotron model path')
    parser.add_argument('-w', '--waveglow_path', type=str, default=None,
                        required=True, help='Waveglow model path')
    parser.add_argument('--hparams', type=str, required=False,
                        help='Comma separated name=value pairs')
    return parser.parse_args()

def plot_data(data, output_path, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', interpolation='none')
    plt.savefig(output_path)
    plt.close(fig)

def load_models(tacotron_path, waveglow_path, hparams_str):
    hparams = create_hparams(hparams_str)
    if hparams_str is None:
        hparams.sampling_rate = 22050

    tacotron_model = load_model(hparams)
    tacotron_model.load_state_dict(torch.load(tacotron_path)['state_dict'])
    tacotron_model.cuda().eval()

    waveglow_model = torch.load(waveglow_path)['model']
    waveglow_model.cuda().eval()
    for k in waveglow_model.convinv:
        k.float()
    
    denoiser = Denoiser(waveglow_model)
    
    return tacotron_model, waveglow_model, denoiser, hparams

def save_wav(file_path, audio_data, sample_rate):
    audio_data = (audio_data * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        audio_data.tobytes(), 
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize, 
        channels=1
    )
    audio_segment.export(file_path, format="wav")

def synthesize_text(tacotron_model, waveglow_model, denoiser, text, hparams, output_directory):
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = tacotron_model.inference(sequence)

    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T),
              os.path.join(output_directory, 'output_plot.png'))

    with torch.no_grad():
        audio = waveglow_model.infer(mel_outputs_postnet, sigma=0.666)
    
    original_audio_path = os.path.join(output_directory, 'output_audio_original.wav')
    audio_data = audio[0].data.cpu().numpy()
    save_wav(original_audio_path, audio_data, hparams.sampling_rate)

    audio_denoised = denoiser(audio, strength=0.01)[:, 0].float()
    denoised_audio_path = os.path.join(output_directory, 'output_audio_denoised.wav')
    audio_denoised_data = audio_denoised.data.cpu().numpy()
    save_wav(denoised_audio_path, audio_denoised_data, hparams.sampling_rate)

    print(f'Plots and audio saved in {output_directory}')

def save_metadata(args, output_directory):
    metadata_path = os.path.join(output_directory, 'metadata.txt')
    with open(metadata_path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
    print(f'Metadata saved in {metadata_path}')

if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_subdir = os.path.join(args.output_directory, timestamp)
    os.makedirs(output_subdir)

    save_metadata(args, output_subdir)

    tacotron_model, waveglow_model, denoiser, hparams = load_models(args.tacotron_path, args.waveglow_path, args.hparams)
    
    synthesize_text(tacotron_model, waveglow_model, denoiser, args.input_text, hparams, output_subdir)