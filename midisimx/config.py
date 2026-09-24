#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#	midisimx Config Python Module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2026
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
###################################################################################
###################################################################################
#
#   Copyright 2026 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
'''

###################################################################################

x_transformers_version = '2.3.1'

tmidix_version = '26.9.23'

torch_version = '2.7.0'

###################################################################################

repos = {
    'hugging_face': {
        'models': {
            'repo_name': 'projectlosangeles/midisimx',
            'repo_type': 'model'
            },        
        'embeddings': {
            'repo_name': 'projectlosangeles/midisimx-embeddings',
            'repo_type': 'dataset'
            },        
        'output_samples': {
            'repo_name': 'projectlosangeles/midisimx-samples',
            'repo_type': 'dataset'
            },
        'midi_dataset': {
            'repo_name': 'projectlosangeles/Discover-MIDI-Dataset',
            'repo_type': 'dataset'
            },        
        'train_data': {
            'repo_name': 'asigalov61/Discover-Piano',
            'repo_type': 'dataset'
            }
        },
    'github': {
        'repo_name': 'asigalov61/midisimx',
        'repo_type': 'code'
       }
    }

###################################################################################

models = {
    'base_encoder': {
        'dim': 768,
        'depth': 16,
        'heads': 12,
        'max_seq_len': 3072,
        'pad_idx': 719,
        'checkpoint': 'midisimx_trained_model_14391_steps_0.255_loss_0.9036_acc.pth'
        },
    'tiny_encoder': {
        'dim': 128,
        'depth': 24,
        'heads': 4,
        'max_seq_len': 3072,
        'pad_idx': 719,
        'checkpoint': 'midisimx_tiny_trained_model_14401_steps_0.5146_loss_0.8202_acc.pth'
        },
    'drums_encoder': {
        'dim': 768,
        'depth': 16,
        'heads': 12,
        'max_seq_len': 1280,
        'mask_prob': 0.15,
        'mask_idx': 821,
        'pad_idx': 822,
        'vocab_size': 823,
        'num_style_classes': 20,
        'num_bpm_classes': 32,
        'checkpoint': 'midisimx_drums_encoder_trained_model_6417_steps_0.6751_loss_0.7739_acc.pth'
        },
    'drums_style_cls': {
        'dim': 384,
        'depth': 8,
        'heads': 8,
        'max_seq_len': 1024,
        'num_classes': 18,
        'use_cls_token': False,
        'average_pool_embed': True,
        'emb_dropout': 0.2,
        'layer_dropout': 0.2,
        'attn_dropout': 0.2,
        'ff_dropout': 0.2,
        'pad_idx': 768,
        'vocab_size': 769,
        'checkpoint': 'midisimx_drums_style_cls_trained_model_13914_steps_0.2076_loss_0.9347_acc.pth'
        },
    'drums_bpm_cls': {
        'dim': 384,
        'depth': 8,
        'heads': 8,
        'max_seq_len': 1024,
        'num_classes': 30,
        'use_cls_token': False,
        'average_pool_embed': True,
        'emb_dropout': 0.2,
        'layer_dropout': 0.2,
        'attn_dropout': 0.2,
        'ff_dropout': 0.2,
        'pad_idx': 768,
        'vocab_size': 769,
        'checkpoint': 'midisimx_drums_bpm_cls_trained_model_6956_steps_0.267_loss_0.9271_acc.pth'
        }
    }

###################################################################################

midi_encoding = {
    'base_encoder': {
        'delta_start_times': list(range(0, 128)),
        'pitches': list(range(128, 256)),
        'durations': list(range(256, 384)),
        'notes': list(range(384, 396)),
        'chords': list(range(396, 717))
        },
    'tiny_encoder': {
        'delta_start_times': list(range(0, 128)),
        'pitches': list(range(128, 256)),
        'durations': list(range(256, 384)),
        'notes': list(range(384, 396)),
        'chords': list(range(396, 717))
        },
    'drums_encoder': {
        'sos': [820],
        'delta_start_times': list(range(0, 256)),
        'pitches': list(range(256, 384)),
        'durations': list(range(384, 640)),
        'velocities': list(range(640, 768)),
        'style': list(range(768, 788)),
        'bpm': list(range(788, 820))
        },
    'drums_style_cls': {
        'num_classes': 18,
        'delta_start_times': list(range(0, 256)),
        'pitches': list(range(256, 384)),
        'durations': list(range(384, 640)),
        'velocities': list(range(640, 768)),
        },
    'drums_bpm_cls': {
        'num_classes': 30,
        'delta_start_times': list(range(0, 256)),
        'pitches': list(range(256, 384)),
        'durations': list(range(384, 640)),
        'velocities': list(range(640, 768)),
        }
    }

###################################################################################
        
drums_style_to_id = {
    'afrobeat': 0,
    'afrocuban': 1,
    'blues': 2,
    'country': 3,
    'dance': 4,
    'funk': 5,
    'gospel': 6,
    'highlife': 7,
    'hiphop': 8,
    'jazz': 9,
    'latin': 10,
    'middleeastern': 11,
    'neworleans': 12,
    'pop': 13,
    'punk': 14,
    'reggae': 15,
    'rock': 16,
    'soul': 17,
    'none': 18,
    'unknown': 19
}

drums_bpm_to_id = {
    50: 0,
    60: 1,
    65: 2,
    70: 3,
    75: 4,
    80: 5,
    85: 6,
    90: 7,
    95: 8,
    100: 9,
    105: 10,
    110: 11,
    115: 12,
    120: 13,
    125: 14,
    130: 15,
    135: 16,
    140: 17,
    145: 18,
    150: 19,
    155: 20,
    160: 21,
    170: 22,
    175: 23,
    180: 24,
    185: 25,
    190: 26,
    200: 27,
    215: 28,
    290: 29,
    'none': 30,
    'unknown': 31
}

id_to_drums_style = {v: k for k, v in drums_style_to_id.items()}

id_to_drums_bpm = {v: k for k, v in drums_bpm_to_id.items()}

###################################################################################

full_config = {
    'repos': repos,
    'models': models,
    'midi_encoding': midi_encoding,
    'drums_style_to_id': drums_style_to_id,
    'id_to_drums_style': id_to_drums_style,
    'drums_bpm_to_id': drums_bpm_to_id,
    'id_to_drums_bpm': id_to_drums_bpm
    }

###################################################################################
# This is the end of the midisimx Config Python Module
###################################################################################