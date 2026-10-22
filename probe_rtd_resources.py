"""Budgeted synthetic ModernBERT compute probe, never fits competition labels."""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'.artifact_runtime/rtd_preflight_20261022'
sys.path[:0]=[str(ROOT/'.artifact_runtime/tabicl_finetune_deps'),
              str(ROOT/'.artifact_runtime/tabicl_runtime')]
sys.stdout.reconfigure(encoding='utf-8')
P=json.loads((RUN/'protocol.json').read_text('utf-8'))

def save(name,data):
    (RUN/name).write_text(json.dumps(data,indent=2),encoding='utf-8')

def guard():
    if time.time()>P['deadline_epoch']:raise TimeoutError('Cumulative 900-second budget expired')

def main():
    guard()
    import torch
    import torch.nn.functional as F
    from transformers import ModernBertConfig,ModernBertModel
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(20261022)
    cfg=ModernBertConfig.from_dict(json.loads((RUN/'upstream_config.json').read_text('utf-8')))
    cfg.reference_compile=False
    cfg._attn_implementation='sdpa'
    cfg._attn_implementation_internal='sdpa'
    model=ModernBertModel(cfg).cuda()
    head=torch.nn.Linear(768,1).cuda()
    # Synthetic learned embeddings stand in for the tokenizer. This omits its cost.
    tokenbank=torch.nn.Embedding(4096,768).cuda()
    params=[*model.parameters(),*head.parameters(),*tokenbank.parameters()]
    opt=torch.optim.AdamW(params,lr=5e-5,fused=True)
    config_check={'layers':cfg.num_hidden_layers,'width':cfg.hidden_size,
        'intermediate':cfg.intermediate_size,'heads':cfg.num_attention_heads,
        'total_backbone_parameters':sum(p.numel() for p in model.parameters()),
        'synthetic_tokenizer_parameters':sum(p.numel() for p in tokenbank.parameters()),
        'attention_implementation':model.config._attn_implementation,
        'reference_compile':cfg.reference_compile,'weight_initialization':'random; no pretrained weights downloaded',
        'torch':torch.__version__,'device':torch.cuda.get_device_name(0)}
    save('configuration.json',config_check)
    results=[]
    for bs in P['batch_probe']:
        guard()
        print('Probing batch',bs,flush=True)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            ids=torch.randint(0,4096,(bs,24),device='cuda')
            pos=torch.zeros((bs,24),device='cuda',dtype=torch.long)
            targets=(torch.rand((bs,21),device='cuda')<.15).float()
            times=[]
            model.train()
            for step in range(10):
                guard();opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize();start=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden=model(inputs_embeds=tokenbank(ids),position_ids=pos).last_hidden_state
                    logits=head(hidden[:,1:22]).squeeze(-1)
                    loss=F.binary_cross_entropy_with_logits(logits.float(),targets)
                assert bool(torch.isfinite(loss))
                loss.backward()
                torch.cuda.synchronize();duration=time.perf_counter()-start
                if step>=2:times.append(duration)
                # Initialize optimizer state and verify actual updates on synthetic data.
                if step in [0,9]:opt.step()
            torch.cuda.synchronize();start=time.perf_counter();opt.step();torch.cuda.synchronize()
            optimizer_seconds=time.perf_counter()-start
            micros=sorted(times)
            median=(micros[3]+micros[4])/2
            rtd_rows=P['pretrain_rows']*P['pretrain_epochs']
            clf_rows=P['supervised_fit_rows']*P['supervised_epochs']*P['supervised_arms']
            total_rows=rtd_rows+clf_rows
            projection=total_rows/bs*median+math.ceil(total_rows/P['effective_batch'])*optimizer_seconds
            item={'batch':bs,'microstep_seconds':times,'median_microstep_seconds':median,
                  'microstep_rows_per_second':bs/median,'optimizer_seconds':optimizer_seconds,
                  'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                  'projected_training_seconds':projection,'pretrain_rows_total':rtd_rows,
                  'supervised_rows_total':clf_rows,'projection_excludes':'real tokenizer, auxiliary features, data preparation, stop/evaluation inference, checkpointing, verification; classification cost approximated by RTD backbone cost',
                  'synthetic_final_loss':float(loss)}
            results.append(item);print(json.dumps(item),flush=True)
        except torch.cuda.OutOfMemoryError as e:
            opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            results.append({'batch':bs,'error':'CUDA out of memory','detail':str(e)[:300]})
        save('probe_results.json',results)
    successful=[x for x in results if 'projected_training_seconds'in x]
    if not successful:decision='STOP_NO_BATCH_FIT'
    else:
        best=min(x['projected_training_seconds'] for x in successful)
        decision='STOP_COST_GATE' if best>max(0,P['deadline_epoch']-time.time()) else 'COST_GATE_PASSED_REQUIRES_INPUT_IMPLEMENTATION'
    preserved={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in P['preserved'].items()}
    save('decision.json',{'decision':decision,'elapsed_seconds':time.time()-P['start_epoch'],
        'remaining_seconds':P['deadline_epoch']-time.time(),'preserved':preserved,
        'competition_labels_read':False,'real_data_training_started':False,'candidate_auc_computed':False,
        'new_submission_generated':False})
    print(decision,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        save('error.json',{'type':type(e).__name__,'error':str(e),'elapsed_seconds':time.time()-P['start_epoch']})
        raise
