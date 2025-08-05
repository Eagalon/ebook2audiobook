import hashlib
import math
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
import wave

import numpy as np
import regex as re
import soundfile as sf
import torch
import torchaudio

from huggingface_hub import hf_hub_download
from pathlib import Path
from pprint import pprint

from lib import *
from lib.classes.tts_engines.common.utils import unload_tts, append_sentence2vtt
from lib.classes.tts_engines.common.audio_filters import detect_gender, trim_audio, normalize_audio, is_audio_data_valid

lock = threading.Lock()

class Piper:

    def __init__(self, session):
        try:
            self.session = session
            self.cache_dir = tts_dir
            self.speakers_path = None
            self.tts_key = f"{self.session['tts_engine']}-{self.session['fine_tuned']}"
            self.sentences_total_time = 0.0
            self.sentence_idx = 1
            self.params = {TTS_ENGINES['PIPER']: {}}
            self.params[self.session['tts_engine']]['samplerate'] = models[self.session['tts_engine']][self.session['fine_tuned']]['samplerate']
            self.vtt_path = os.path.join(self.session['process_dir'], os.path.splitext(self.session['final_name'])[0] + '.vtt')    
            self.resampler_cache = {}
            self.audio_segments = []
            self._build()
        except Exception as e:
            error = f'__init__() error: {e}'
            print(error)
            return None

    def _build(self):
        try:
            tts = (loaded_tts.get(self.tts_key) or {}).get('engine', False)
            if not tts:
                if self.session['tts_engine'] == TTS_ENGINES['PIPER']:
                    if self.session['custom_model'] is not None:
                        msg = f"{self.session['tts_engine']} custom model not implemented yet!"
                        print(msg)
                        return False
                    else:
                        model_path = models[self.session['tts_engine']][self.session['fine_tuned']]['repo']
                        tts = self._load_api(self.tts_key, model_path, self.session['device'])
            return (loaded_tts.get(self.tts_key) or {}).get('engine', False)
        except Exception as e:
            error = f'build() error: {e}'
            print(error)
            return False

    def _load_api(self, key, model_path, device):
        global lock
        try:
            if key in loaded_tts.keys():
                return loaded_tts[key]['engine']
            unload_tts(device, [self.tts_key])
            with lock:
                from piper import PiperVoice
                
                # Get voice model name from session or use language-based default
                voice_name = self.session.get('voice') or self.session.get('voice_model')
                
                # If no voice specified, use language-based selection
                if not voice_name:
                    language_voice_map = {
                        'eng': 'en_US-lessac-medium',
                        'deu': 'de_DE-thorsten-medium',
                        'fra': 'fr_FR-upmc-medium',
                        'spa': 'es_ES-davefx-medium',
                        'ita': 'it_IT-riccardo-x_low',
                        'por': 'pt_BR-edresson-low'
                    }
                    language = self.session.get('language', 'eng')
                    voice_name = language_voice_map.get(language, 'en_US-lessac-medium')
                
                # Validate voice name
                if voice_name not in default_engine_settings[TTS_ENGINES['PIPER']]['voices']:
                    voice_name = 'en_US-lessac-medium'  # fallback
                
                # Map voice names to their subdirectory paths in rhasspy/piper-voices
                voice_paths = {
                    'en_US-lessac-medium': 'en/en_US/lessac/medium',
                    'en_US-amy-medium': 'en/en_US/amy/medium',
                    'en_GB-alba-medium': 'en/en_GB/alba/medium',
                    'en_GB-aru-medium': 'en/en_GB/aru/medium',
                    'de_DE-thorsten-medium': 'de/de_DE/thorsten/medium',
                    'fr_FR-upmc-medium': 'fr/fr_FR/upmc/medium',
                    'es_ES-davefx-medium': 'es/es_ES/davefx/medium',
                    'it_IT-riccardo-x_low': 'it/it_IT/riccardo-x_low/x_low',
                    'pt_BR-edresson-low': 'pt/pt_BR/edresson/low'
                }
                
                if voice_name not in voice_paths:
                    print(f"Unknown voice model: {voice_name}, using default")
                    voice_name = 'en_US-lessac-medium'
                
                voice_path = voice_paths[voice_name]
                
                model_file = hf_hub_download(
                    repo_id=model_path,
                    filename=f"{voice_path}/{voice_name}.onnx",
                    cache_dir=self.cache_dir
                )
                config_file = hf_hub_download(
                    repo_id=model_path,
                    filename=f"{voice_path}/{voice_name}.onnx.json",
                    cache_dir=self.cache_dir
                )
                
                use_cuda = device == 'cuda' and torch.cuda.is_available()
                tts = PiperVoice.load(model_file, config_path=config_file, use_cuda=use_cuda)
                
                if tts:
                    loaded_tts[key] = {"engine": tts, "config": None} 
                    msg = f'{model_path} ({voice_name}) Loaded!'
                    print(msg)
                    return tts
                else:
                    error = 'TTS engine could not be created!'
                    print(error)
        except Exception as e:
            error = f'_load_api() error: {e}'
            print(error)
        return False
        
    def _tensor_type(self, audio_data):
        if isinstance(audio_data, torch.Tensor):
            return audio_data
        elif isinstance(audio_data, np.ndarray):  
            return torch.from_numpy(audio_data).float()
        elif isinstance(audio_data, list):  
            return torch.tensor(audio_data, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported type for audio_data: {type(audio_data)}")
            
    def _get_resampler(self, orig_sr, target_sr):
        key = (orig_sr, target_sr)
        if key not in self.resampler_cache:
            self.resampler_cache[key] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=target_sr
            )
        return self.resampler_cache[key]

    def convert(self, sentence_number, sentence):
        try:
            audio_data = False
            trim_audio_buffer = 0.004
            settings = self.params[self.session['tts_engine']]
            final_sentence_file = os.path.join(self.session['chapters_dir_sentences'], f'{sentence_number}.{default_audio_proc_format}')
            sentence = sentence.strip()
            
            tts = (loaded_tts.get(self.tts_key) or {}).get('engine', False)
            if tts:
                if sentence == TTS_SML['break']:
                    break_tensor = torch.zeros(1, int(settings['samplerate'] * (int(np.random.uniform(0.3, 0.6) * 100) / 100))) # 0.3 to 0.6 seconds
                    self.audio_segments.append(break_tensor.clone())
                    return True
                elif sentence == TTS_SML['pause']:
                    pause_tensor = torch.zeros(1, int(settings['samplerate'] * (int(np.random.uniform(1.0, 1.8) * 100) / 100))) # 1.0 to 1.8 seconds
                    self.audio_segments.append(pause_tensor.clone())
                    return True
                else:
                    if self.session['tts_engine'] == TTS_ENGINES['PIPER']:
                        # Generate audio using Piper
                        try:
                            # Create a temporary WAV file for Piper output
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                                temp_wav_path = temp_wav.name
                            
                            try:
                                # Use Piper's synthesize method to generate audio directly to WAV file
                                with wave.open(temp_wav_path, 'wb') as wav_file:
                                    tts.synthesize(sentence, wav_file)
                                
                                # Read the WAV file and extract audio data
                                with wave.open(temp_wav_path, 'rb') as wav_file:
                                    frames = wav_file.getnframes()
                                    sample_rate = wav_file.getframerate()
                                    sample_width = wav_file.getsampwidth()
                                    channels = wav_file.getnchannels()
                                    
                                    if frames == 0:
                                        error = 'Generated WAV file contains no audio frames'
                                        print(error)
                                        audio_sentence = None
                                    else:
                                        # Read the raw audio data
                                        audio_data = wav_file.readframes(frames)
                                        
                                        # Convert raw audio data to numpy array
                                        if sample_width == 2:
                                            # 16-bit audio (most common for Piper)
                                            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                                            audio_array = audio_array / 32768.0
                                        else:
                                            error = f'Unsupported sample width: {sample_width}'
                                            print(error)
                                            audio_sentence = None
                                            
                                        if audio_array is not None:
                                            # Handle multi-channel audio by taking only the first channel
                                            if channels > 1:
                                                audio_array = audio_array.reshape(-1, channels)[:, 0]
                                            
                                            audio_sentence = audio_array
                                            
                            finally:
                                # Clean up temporary file
                                if os.path.exists(temp_wav_path):
                                    os.unlink(temp_wav_path)
                                    
                        except Exception as e:
                            error = f'Error synthesizing with Piper: {e}'
                            print(error)
                            audio_sentence = None
                            
                    if is_audio_data_valid(audio_sentence):
                        sourceTensor = self._tensor_type(audio_sentence)
                        audio_tensor = sourceTensor.clone().detach().unsqueeze(0).cpu()
                        if sentence[-1].isalnum():
                            audio_tensor = trim_audio(audio_tensor.squeeze(), settings['samplerate'], 0.003, trim_audio_buffer).unsqueeze(0)
                        self.audio_segments.append(audio_tensor)
                        if not re.search(r'\w$', sentence, flags=re.UNICODE):
                            break_tensor = torch.zeros(1, int(settings['samplerate'] * (int(np.random.uniform(0.3, 0.6) * 100) / 100)))
                            self.audio_segments.append(break_tensor.clone())
                        if self.audio_segments:
                            audio_tensor = torch.cat(self.audio_segments, dim=-1)
                            start_time = self.sentences_total_time
                            duration = audio_tensor.shape[-1] / settings['samplerate']
                            end_time = start_time + duration
                            self.sentences_total_time = end_time
                            sentence_obj = {
                                "start": start_time,
                                "end": end_time,
                                "text": sentence,
                                "resume_check": self.sentence_idx
                            }
                            self.sentence_idx = append_sentence2vtt(sentence_obj, self.vtt_path)
                            if self.sentence_idx:
                                torchaudio.save(final_sentence_file, audio_tensor, settings['samplerate'], format=default_audio_proc_format)
                                del audio_tensor
                        self.audio_segments = []
                        if os.path.exists(final_sentence_file):
                            return True
                        else:
                            error = f"Cannot create {final_sentence_file}"
                            print(error)
            else:
                error = f"convert() error: {self.session['tts_engine']} is None"
                print(error)
        except Exception as e:
            error = f'Piper.convert(): {e}'
            raise ValueError(e)
        return False