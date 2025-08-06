import hashlib
import math
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
import io

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
                        # Get voice name from session
                        voice_name = self.session.get('voice') or self.session.get('voice_model', 'en_US-lessac-medium')
                        
                        # Map language to default voice if needed
                        if voice_name == 'en_US-lessac-medium' and self.session.get('language') in ['fra', 'deu', 'spa', 'ita', 'por']:
                            language_voice_map = {
                                'fra': 'fr_FR-upmc-medium',
                                'deu': 'de_DE-thorsten-medium', 
                                'spa': 'es_ES-davefx-medium',
                                'ita': 'it_IT-riccardo-x_low',
                                'por': 'pt_BR-edresson-low'
                            }
                            voice_name = language_voice_map.get(self.session['language'], voice_name)
                        
                        tts = self._load_piper_model(self.tts_key, voice_name, self.session['device'])
            return (loaded_tts.get(self.tts_key) or {}).get('engine', False)
        except Exception as e:
            error = f'build() error: {e}'
            print(error)
            return False

    def _load_piper_model(self, key, voice_name, device):
        global lock
        try:
            if key in loaded_tts.keys():
                return loaded_tts[key]['engine']
            unload_tts(device, [self.tts_key])
            with lock:
                try:
                    from piper import PiperVoice
                    
                    # Download voice model from HuggingFace
                    repo_id = models[TTS_ENGINES['PIPER']]['internal']['repo']
                    
                    # Get model files based on voice name
                    model_file = f"{voice_name}.onnx"
                    config_file = f"{voice_name}.onnx.json"
                    
                    # Download the model files
                    model_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=model_file,
                        cache_dir=self.cache_dir
                    )
                    config_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=config_file,
                        cache_dir=self.cache_dir
                    )
                    
                    # Load the Piper voice
                    tts = PiperVoice.load(model_path, config_path)
                    
                    if tts:
                        loaded_tts[key] = {"engine": tts, "config": None, "voice_name": voice_name}
                        msg = f'PIPER voice {voice_name} loaded!'
                        print(msg)
                        return tts
                    else:
                        error = 'PIPER TTS engine could not be created!'
                        print(error)
                except ImportError:
                    error = 'PIPER TTS not installed. Please install with: pip install piper-tts'
                    print(error)
                except Exception as e:
                    error = f'Failed to load PIPER voice {voice_name}: {e}'
                    print(error)
        except Exception as e:
            error = f'_load_piper_model() error: {e}'
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
            settings = self.params[self.session['tts_engine']]
            final_sentence_file = os.path.join(self.session['chapters_dir_sentences'], f'{sentence_number}.{default_audio_proc_format}')
            sentence = sentence.strip()
            
            tts = (loaded_tts.get(self.tts_key) or {}).get('engine', False)
            if tts:
                if sentence[-1].isalnum():
                    sentence = f'{sentence} —'
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
                        # Generate audio using PIPER TTS
                        audio_arrays = []
                        for audio_chunk in tts.synthesize(sentence):
                            # Use the audio_float_array property for proper audio synthesis
                            audio_arrays.append(audio_chunk.audio_float_array)
                        
                        if audio_arrays:
                            # Concatenate all audio chunks
                            audio_np = np.concatenate(audio_arrays)
                            audio_sentence = audio_np
                        else:
                            error = f"PIPER failed to generate audio for: {sentence}"
                            print(error)
                            return False
                            
                    if is_audio_data_valid(audio_sentence):
                        sourceTensor = self._tensor_type(audio_sentence)
                        audio_tensor = sourceTensor.clone().detach().unsqueeze(0).cpu()
                        if sentence[-1].isalnum() or sentence[-1] == '—':
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