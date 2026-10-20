import json
from pathlib import Path

import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, SiglipVisionConfig, SiglipVisionModel

from mibot.data.recap_language import IM_END_ID, IMAGE_ID, insert_guidance
from mibot.models.recap_rlinf import build_critic, verify_upstream


def test_packed_language_guidance_updates_positions_and_context():
    inputs=dict(input_ids=torch.tensor([[10,IMAGE_ID,20,IM_END_ID,30,40,10,IMAGE_ID,20,IM_END_ID,30]]),
        position_ids=torch.cat([torch.arange(6),torch.arange(5)]).view(1,1,-1).repeat(3,1,1),
        cu_seq_lens_q=torch.tensor([0,6,11],dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0,6,11],dtype=torch.int32))
    result,segments=insert_guidance(inputs,torch.tensor([True,False]),torch.tensor([51,52]),torch.tensor([[0,4],[6,10]]))
    assert result['input_ids'].tolist()==[[10,IMAGE_ID,20,51,52,IM_END_ID,30,40,10,IMAGE_ID,20,IM_END_ID,30]]
    assert result['cu_seq_lens_q'].tolist()==[0,8,13]
    assert segments.tolist()==[[0,6],[8,12]]
    torch.testing.assert_close(result['position_ids'][0,0],torch.cat([torch.arange(8),torch.arange(5)]))
    assert inputs['input_ids'].shape[-1]==11


def test_deployment_language_guidance_preserves_padding_and_image_tokens():
    inputs=dict(input_ids=torch.tensor([[10,IMAGE_ID,20,IM_END_ID,30],[10,IMAGE_ID,IM_END_ID,30,0]]),
                attention_mask=torch.tensor([[1,1,1,1,1],[1,1,1,1,0]]))
    result,_=insert_guidance(inputs,torch.tensor([True,True]),torch.tensor([51,52]))
    assert result['input_ids'].tolist()==[[10,IMAGE_ID,20,51,52,IM_END_ID,30],[10,IMAGE_ID,51,52,IM_END_ID,30,0]]
    assert result['attention_mask'].sum(-1).tolist()==[7,6]


def test_native_critic_backpropagates_through_pretrained_paths(tmp_path):
    torch.set_num_threads(1)
    assert len(verify_upstream())==40
    siglip=tmp_path/'siglip';gemma=tmp_path/'gemma'
    SiglipVisionModel(SiglipVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=4,image_size=224,patch_size=14)).save_pretrained(siglip)
    Gemma3ForCausalLM(Gemma3TextConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=4,
        num_attention_heads=1,num_key_value_heads=1,head_dim=256,vocab_size=128,
        sliding_window=2048,max_position_embeddings=2048)).save_pretrained(gemma)
    model=build_critic(siglip,gemma).float().train()
    observation=dict(images={'base_0_rgb':torch.rand(2,3,224,224)*2-1},
        tokenized_prompt=torch.randint(0,128,(2,8)),tokenized_prompt_mask=torch.ones(2,8,dtype=torch.bool))
    result=model(observation,target_values=torch.tensor([-.2,-.8]))
    assert result.logits.shape==(2,201)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    for p in [model.value_expert.vision_tower.vision_model.embeddings.patch_embedding.weight,
              model.value_expert.gemma3.model.embed_tokens.weight,
              model.value_expert.multi_modal_proj.weight,model.value_head.cls_embedding.weight,
              model.value_head.value_proj.weight]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0
    assert ((result.predicted_values>=-1)&(result.predicted_values<=0)).all()
