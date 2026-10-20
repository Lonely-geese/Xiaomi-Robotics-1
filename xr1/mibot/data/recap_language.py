"""RLinf's positive language guidance for packed and unpacked XR1 inputs."""
import torch


GUIDANCE_TEXT = '\nAdvantage: positive'
IM_END_ID = 151645
IMAGE_ID = 151655


def insert_guidance(inputs, flags, suffix, segments=None):
    """Insert guidance in the image/task turn and update packing/positions."""
    result = dict(inputs)
    packed = 'cu_seq_lens_q' in inputs
    tokens = inputs['input_ids']
    if packed:
        offsets = inputs['cu_seq_lens_q'].tolist()
        sequences = [tokens[0,a:b] for a,b in zip(offsets[:-1], offsets[1:])]
    else:
        mask = inputs.get('attention_mask', torch.ones_like(tokens))
        lengths = mask.sum(-1).tolist()
        if not torch.equal(mask.bool(), torch.arange(tokens.shape[1],device=tokens.device)[None] < mask.sum(-1)[:,None]):
            raise ValueError('RECAP language guidance requires right-padded inputs')
        sequences = [row[:n] for row,n in zip(tokens,lengths)]
    if len(sequences)!=len(flags):raise ValueError('Guidance flags must match sample count')
    output=[];positions=[];new_segments=[];offset=0
    for i,(seq,flag) in enumerate(zip(sequences,flags)):
        added = len(suffix) if bool(flag) else 0
        insert = 0
        if added:
            images = (seq==IMAGE_ID).nonzero().flatten()
            after = int(images[-1])+1 if len(images) else 0
            ends = (seq[after:]==IM_END_ID).nonzero().flatten()
            if not len(ends):raise ValueError('Cannot locate the end of the task turn')
            insert = after+int(ends[0])
            seq = torch.cat([seq[:insert],suffix.to(seq.device),seq[insert:]])
        output.append(seq)
        if packed:
            a,b=offsets[i:i+2]
            pos=inputs['position_ids'][:,:,a:b]
            if added:
                extra=pos[:,:,insert:insert+1]+torch.arange(added,device=pos.device)[None,None]
                pos=torch.cat([pos[:,:,:insert],extra,pos[:,:,insert:]+added],dim=-1)
            positions.append(pos)
            if segments is not None:
                start,end=segments[i].tolist()
                if added and insert>=end-a:raise ValueError('Guidance fell outside action-conditioning context')
                new_segments.append([offset+start-a,offset+end-a+added])
        offset+=len(seq)
    if packed:
        lengths=[len(s) for s in output]
        result['input_ids']=torch.cat(output)[None]
        result['position_ids']=torch.cat(positions,dim=-1)
        cumulative=torch.tensor([0]+lengths,device=tokens.device,dtype=torch.int32).cumsum(0,dtype=torch.int32)
        result.update(cu_seq_lens_q=cumulative,cu_seq_lens_k=cumulative,
                      max_length_q=max(lengths),max_length_k=max(lengths))
        return result,torch.tensor(new_segments,device=tokens.device,dtype=torch.long) if segments is not None else None
    result['input_ids']=torch.nn.utils.rnn.pad_sequence(output,batch_first=True,padding_value=0)
    result['attention_mask']=(torch.arange(result['input_ids'].shape[1],device=tokens.device)[None]
                              < torch.tensor([len(s) for s in output],device=tokens.device)[:,None]).to(mask.dtype)
    result.pop('position_ids',None)
    return result,None
