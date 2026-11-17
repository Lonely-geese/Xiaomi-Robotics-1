"""Compare optimized input batching against the official XR1 input pipeline."""
from pathlib import Path
import argparse,os,sys,json
import numpy as np
os.environ.setdefault('XR1_USB_PREPARED_DIR',str(Path(__file__).resolve().parents[2]/'prepared_gripper_v2'))
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mibot.usb1175 import USB1175Dataset,CachedCollate,build_templates,PREPARED,array,prepared_path
from mibot.data.datasets.json_dataset import JsonDataset
from mibot.data.collate.custom_collate import CustomCollate
from mibot.data.datamodule.base_datamodule import CompleteBatchSampler

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-source',action='store_true',help='Also compare every pose against source_prepared; requires the original prepared dataset')
    args=parser.parse_args()
    torch.set_num_threads(4);os.environ['MAX_LENGTH']='32768';os.environ['XR1_STRICT_BATCH']='1'
    if not (PREPARED/'templates.pt').exists():build_templates()
    train=USB1175Dataset('train');val=USB1175Dataset('val')
    assert len(train)==295704 and len(val)==128
    indices=list(range(31))+[train.rows[0]['length']-1]
    items=[train[i] for i in indices];fast=CachedCollate()(items)
    native=object.__new__(JsonDataset);native.action_length=30;reference=CustomCollate();reference_items=[]
    for index,item in zip(indices,items):
        ep,frame=train.locate(index);row=train.rows[ep]
        with np.load(prepared_path(row['native']),allow_pickle=False) as data:
            traj=dict(num_frames=row['length'],instruction=dict(general=[row['prompt']]))
            for k in data.files:
                section,field=k.split('.',1);traj.setdefault(section,{})[field]=data[k]
        p=native._prompt(traj);steps=min(30,row['length']-frame)
        p['conversations'] += [{'from':'human','value':'Robot state: <state>'},{'from':'gpt','value':''.join(f'<a_{i}>' for i in range(steps))+'<score>'}]
        images=[Image.fromarray(array(path)[frame]) for path in row['cameras']]
        reference_items.append(dict(messages=native._messages(p['conversations'],images),**{k:v for k,v in item.items() if k!='rgb'}))
        left=np.concatenate(native._arm_action(traj,'left',frame,steps),axis=-1)
        decoded=item['action'][:steps,:7].numpy()*(train.std[:steps]+1e-6)+train.mean[:steps]
        valid=item['action_mask'][:steps,:7].numpy()
        np.testing.assert_allclose(decoded[valid],left[:steps][valid],rtol=1e-5,atol=3e-6)
        assert np.all(array(row['states'])[:,7]==np.float32(.09))
        assert not array(row['gripper_mask'])[-3:].any()
    expected=reference(reference_items)
    for key,value in expected.items():
        if key in ['pixel_values','image_grid_thw']:continue
        if torch.is_tensor(value):torch.testing.assert_close(fast[key],value,rtol=0,atol=0)
        else:assert fast[key]==value,key
    assert fast['action'].shape==(32,30,60) and fast['action_mask'][:,:,:7].any() and not fast['action_mask'][:,:,7:].any()
    image=reference.processor.image_processor(images=fast['rgb'].flatten(0,1).permute(0,3,1,2),do_resize=False,return_tensors='pt')
    for key in ['pixel_values','image_grid_thw']:torch.testing.assert_close(image[key],expected[key],rtol=0,atol=0)
    selected=[]
    for rank in range(8):
        sampler=CompleteBatchSampler(train,256,num_replicas=8,rank=rank,shuffle=True,seed=42)
        values=list(iter(sampler));assert len(values)==36960;selected.extend(values)
    assert len(selected)==len(set(selected))==295680
    val_images=[]
    for i in [0,63,127]:
        item=val[i];assert item['action'].shape==(30,60) and item['rgb'].shape==(3,256,256,3)
        val_images.append(i)
    import hashlib
    report=dict(passed=True,gripper_repair_version=2,manifest_sha256=hashlib.sha256((PREPARED/'manifest.json').read_bytes()).hexdigest(),unique_samples_per_gpu=32,world_size=8,global_batch_size=256,gradient_accumulation_steps=1,
        train_frames=len(train),validation_samples=len(val),steps_per_epoch=1155,total_steps=23100,
        complete_global_batches=True,no_cross_rank_duplicates=True,discarded_tail_frames=24,
        token_packing_exact=True,pixel_preprocessing_cpu_exact=True,native_action_targets_verified=True,
        packed_tokens=fast['input_ids'].shape[1],per_image_grid=image['image_grid_thw'][0].tolist())
    verify_dataset_all(train,args.audit_source)
    verify_dataset_all(val,args.audit_source)
    report['source_pose_audited']=args.audit_source
    (PREPARED/'input_verified.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


def verify_dataset_all(dataset,audit_source=False):
    import hashlib
    from mibot.gripper_contract_v2 import anchor_proprio,absolute_gripper
    manifest=json.loads((PREPARED/'manifest.json').read_text())
    before={}
    if audit_source:
        old=json.loads((prepared_path(manifest['source_prepared'])/'manifest.json').read_text());before={r['uid']:r for r in old['rows']}
    for row in dataset.rows:
        target=array(row['actions']);state=array(row['states']);mask=array(row['gripper_mask'])
        assert np.isfinite(target).all() and np.isfinite(state).all() and mask.shape==target.shape[:2]
        for path in row['cameras']:
            rgb=array(path);assert rgb.shape==(row['length'],256,256,3) and rgb.dtype==np.uint8
        if audit_source:
            original=before[row['uid']]
            assert row['cameras']==original['cameras'],'Source audit requires the original RGB paths'
            np.testing.assert_array_equal(target[:,:,:6],array(original['actions'])[:,:,:6])
            keep=np.arange(60)!=7;np.testing.assert_array_equal(state[:,keep],array(original['states'])[:,keep])
        assert np.all(state[:,7]==np.float32(.09)) and not mask[-3:].any()
        # Model output is an absolute command represented relative to a fixed anchor.
        contract=json.loads((PREPARED/'gripper_contract.json').read_text())
        if 'gripper_labels' in row:
            with np.load(prepared_path(row['gripper_labels'])) as labels:
                closed=contract['closed_width_m']
                for offset in [0,1,29]:
                    idx=np.minimum(np.arange(row['length'])+offset,row['length']-1)
                    recovered=absolute_gripper(target[:,offset],closed)
                    np.testing.assert_allclose(recovered,labels['action_width'][idx],atol=1e-8,rtol=0)
        np.testing.assert_array_equal(anchor_proprio(state),state)
    return True

if __name__=='__main__':main()
