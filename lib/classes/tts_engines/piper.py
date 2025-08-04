import os
import re
import threading
import tempfile
import wave
import numpy as np
import torch
import torchaudio

from pathlib import Path

from lib import *

lock = threading.Lock()

class Piper:

    def __init__(self, session):
        try:
            self.session = session
            self.cache_dir = tts_dir
            self.tts_key = f"{self.session['tts_engine']}-{self.session['fine_tuned']}"
            self.voice = None
            self.sentences_total_time = 0.0
            self.sentence_idx = 1
            self.params = {TTS_ENGINES['PIPER']: {}}
            self.params[self.session['tts_engine']]['samplerate'] = models[self.session['tts_engine']][self.session['fine_tuned']]['samplerate']
            self.vtt_path = os.path.join(self.session['process_dir'], os.path.splitext(self.session['final_name'])[0] + '.vtt')
            self.audio_segments = []
            self._build()
        except Exception as e:
            error = f'Piper.__init__(): {e}'
            print(error)
            raise ValueError(e)

    def _build(self):
        try:
            if self.session['tts_engine'] == TTS_ENGINES['PIPER']:
                if self.session['custom_model'] is not None:
                    # Custom model support - load from custom directory
                    model_dir = os.path.join(self.session['custom_model_dir'], self.session['tts_engine'], self.session['custom_model'])
                    model_file = None
                    config_file = None
                    
                    # Find .onnx and .onnx.json files in the custom model directory
                    for file in os.listdir(model_dir):
                        if file.endswith('.onnx'):
                            model_file = os.path.join(model_dir, file)
                        elif file.endswith('.onnx.json'):
                            config_file = os.path.join(model_dir, file)
                    
                    if not model_file or not config_file:
                        print(f"Piper model files not found in {model_dir}")
                        return False
                    
                    self.tts_key = f"{self.session['tts_engine']}-{self.session['custom_model']}"
                else:
                    # Use default voice model
                    voice_name = self.session.get('voice_model', 'en_US-lessac-medium')
                    if voice_name not in default_engine_settings[TTS_ENGINES['PIPER']]['voices']:
                        voice_name = 'en_US-lessac-medium'  # fallback
                    
                    # Download model files if needed
                    hf_repo = models[self.session['tts_engine']][self.session['fine_tuned']]['repo']
                    
                    try:
                        from huggingface_hub import hf_hub_download
                        model_file = hf_hub_download(
                            repo_id=hf_repo,
                            filename=f"{voice_name}.onnx",
                            cache_dir=self.cache_dir
                        )
                        config_file = hf_hub_download(
                            repo_id=hf_repo,
                            filename=f"{voice_name}.onnx.json",
                            cache_dir=self.cache_dir
                        )
                    except Exception as e:
                        print(f"Failed to download Piper model {voice_name}: {e}")
                        return False

                # Load the Piper voice model
                voice = self._load_piper_voice(model_file, config_file)
                if voice:
                    loaded_tts[self.tts_key] = {
                        'engine': voice,
                        'samplerate': self.params[self.session['tts_engine']]['samplerate']
                    }
                    return True
                else:
                    error = 'Piper TTS engine could not be created!'
                    print(error)
                    return False
            else:
                print(f'Unsupported TTS engine: {self.session["tts_engine"]}')
                return False
        except Exception as e:
            error = f'Piper._build(): {e}'
            print(error)
            return False

    def _load_piper_voice(self, model_file, config_file):
        """Load a Piper voice from model and config files"""
        try:
            from piper import PiperVoice
            
            use_cuda = self.session['device'] == 'gpu' and torch.cuda.is_available()
            voice = PiperVoice.load(model_file, config_path=config_file, use_cuda=use_cuda)
            print(f"Piper voice loaded successfully from {model_file}")
            return voice
        except ImportError:
            error = 'piper-tts package not installed. Please install with: pip install piper-tts'
            print(error)
            return None
        except Exception as e:
            error = f'Error loading Piper voice: {e}'
            print(error)
            return None

    def convert(self, sentence_number, sentence):
        try:
            # Import needed functions when actually used
            from lib.classes.tts_engines.common.utils import append_sentence2vtt
            from lib.classes.tts_engines.common.audio_filters import trim_audio, is_audio_data_valid
            
            settings = self.params[self.session['tts_engine']]
            final_sentence_file = os.path.join(self.session['chapters_dir_sentences'], f'{sentence_number}.{default_audio_proc_format}')
            sentence = sentence.strip()

            # Get the loaded Piper voice
            voice = (loaded_tts.get(self.tts_key) or {}).get('engine', False)
            if not voice:
                error = f"Piper voice not loaded for key: {self.tts_key}"
                print(error)
                return False

            # Handle special tokens
            if sentence == TTS_SML['break']:
                break_tensor = torch.zeros(1, int(settings['samplerate'] * (int(np.random.uniform(0.3, 0.6) * 100) / 100)))
                self.audio_segments.append(break_tensor.clone())
                return True
            elif sentence == TTS_SML['pause']:
                pause_tensor = torch.zeros(1, int(settings['samplerate'] * (int(np.random.uniform(1.0, 1.8) * 100) / 100)))
                self.audio_segments.append(pause_tensor.clone())
                return True

            # Generate audio using Piper
            audio_data = self._synthesize_with_piper(voice, sentence)
            
            if is_audio_data_valid(audio_data):
                # Convert to tensor and process
                audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
                
                # Trim audio if sentence ends with alphanumeric character
                if sentence[-1].isalnum():
                    audio_tensor = trim_audio(audio_tensor.squeeze(), settings['samplerate'], 0.003, 0.004).unsqueeze(0)
                
                self.audio_segments.append(audio_tensor)
                
                # Add break after non-word endings
                if not re.search(r'\w$', sentence, flags=re.UNICODE):
                    break_tensor = torch.zeros(1, int(settings['samplerate'] * (int(np.random.uniform(0.3, 0.6) * 100) / 100)))
                    self.audio_segments.append(break_tensor.clone())
                
                # Combine audio segments and save
                if self.audio_segments:
                    combined_audio = torch.cat(self.audio_segments, dim=-1)
                    start_time = self.sentences_total_time
                    duration = combined_audio.shape[-1] / settings['samplerate']
                    end_time = start_time + duration
                    self.sentences_total_time = end_time
                    
                    # Create VTT entry
                    sentence_obj = {
                        "start": start_time,
                        "end": end_time,
                        "text": sentence,
                        "resume_check": self.sentence_idx
                    }
                    self.sentence_idx = append_sentence2vtt(sentence_obj, self.vtt_path)
                    
                    if self.sentence_idx:
                        # Save audio file
                        torchaudio.save(final_sentence_file, combined_audio, settings['samplerate'], format=default_audio_proc_format)
                        del combined_audio
                    
                    self.audio_segments = []
                    
                    if os.path.exists(final_sentence_file):
                        return True
                    else:
                        error = f"Cannot create {final_sentence_file}"
                        print(error)
                        return False
            else:
                error = "Invalid audio data generated by Piper"
                print(error)
                return False

        except Exception as e:
            error = f'Piper.convert(): {e}'
            print(error)
            raise ValueError(e)
        
        return False

    def _synthesize_with_piper(self, voice, text):
        """Synthesize audio using Piper voice"""
        try:
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name

            # Synthesize audio to WAV file
            with wave.open(temp_path, 'wb') as wav_file:
                voice.synthesize(text, wav_file)

            # Read the audio data back
            audio_tensor, sample_rate = torchaudio.load(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Convert to numpy array (mono)
            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0)
            else:
                audio_tensor = audio_tensor.squeeze(0)
            
            return audio_tensor.numpy()
            
        except Exception as e:
            error = f'Error synthesizing with Piper: {e}'
            print(error)
            return None