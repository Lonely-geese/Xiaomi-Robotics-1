"""Construct the unmodified RLinf critic and its observation processor."""
import hashlib
import json
from pathlib import Path

from mibot.models.rlinf_recap.recap.configuration import ValueCriticConfig
from mibot.models.rlinf_recap.recap.modeling_critic import ValueCriticModel
from mibot.models.rlinf_recap.recap.processing import ValueImageProcessor, ValueProcessor


IMAGE_KEYS = ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')


def verify_upstream():
    root=Path(__file__).with_name('rlinf_recap')
    manifest=json.loads((root/'upstream.json').read_text())
    for name,entry in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError(f'Native RLinf source was modified: {name}')
    return manifest['commit']


def build_critic(siglip_path,gemma_path):
    verify_upstream()
    config=ValueCriticConfig(siglip_path=str(siglip_path),gemma3_path=str(gemma_path),
        critic_expert_variant='gemma_1m',num_bins=201,v_min=-1.0,v_max=0.0,
        value_dropout=0.0,precision='bfloat16',freeze_vision_encoder=False,
        freeze_vlm=False,stop_gradient_to_vlm=False,max_token_len=200)
    return ValueCriticModel(config)


def build_processor(gemma_path):
    # Images on disk were already resized with this upstream processor's exact
    # uint8 resize_with_pad implementation; augmentation/normalization stay native.
    return ValueProcessor(tokenizer_name_or_path=str(gemma_path),max_token_len=200,
        image_processor=ValueImageProcessor(do_resize=False,do_augment=True))


def observation(images,tokens,mask,processor,training=False):
    views={key:images[:,i] for i,key in enumerate(IMAGE_KEYS)}
    processed=processor.image_processor.process_images(views,train=training)
    return dict(images=processed[0],image_masks=processed[1],
                tokenized_prompt=tokens,tokenized_prompt_mask=mask)
