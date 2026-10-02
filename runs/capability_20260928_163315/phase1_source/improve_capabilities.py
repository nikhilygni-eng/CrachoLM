#!/usr/bin/env python3
"""Bounded supervised continuation of CrachoLM's own weights.

Authored synthetic tasks, grouped train/dev/test splits, answer-only loss,
equal-example loss, balanced categories, saved optimizer, and honest probes.
This is a small curriculum, not a general intelligence benchmark.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from config import Config
from src.inference import load_model_for_inference
from src.chat_response import chat_prompt, generate_chat_reply
from capability_curriculum import make_curriculum, normalize

ROOT = Path(__file__).resolve().parent
CATEGORY_WEIGHTS = dict(grammar=.25, arithmetic=.22, reading=.20,
                        explanation=.23, comparison=.04, chat=.06)


def encode_example(row, tokenizer, max_length=192):
    prefix = tokenizer.encode(chat_prompt(row['prompt']), add_special_tokens=False)
    answer = tokenizer.encode(row['answer'], add_special_tokens=False) + [tokenizer.eos_id]
    ids = prefix + answer
    if len(ids)-1 > max_length:
        raise ValueError('Example exceeds context; never silently truncate its answer.')
    if tokenizer.unk_id in ids:
        raise ValueError('Unknown tokens in training example.')
    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
    y[:len(prefix)-1] = 0
    return x, y


def collate(batch):
    length = max(len(x) for x,y in batch)
    xs = torch.zeros((len(batch), length), dtype=torch.long)
    ys = torch.zeros_like(xs)
    for i,(x,y) in enumerate(batch):
        xs[i,:len(x)], ys[i,:len(y)] = x,y
    return xs,ys


class Examples(Dataset):
    def __init__(self, rows, tok, max_length):
        self.rows = rows
        self.data = [encode_example(row,tok,max_length) for row in rows]
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        return self.data[i]


def example_loss(logits, targets):
    losses = F.cross_entropy(logits.transpose(1,2),targets,ignore_index=0,reduction='none')
    counts = (targets != 0).sum(1)
    if torch.any(counts == 0):
        raise ValueError('Example has no answer targets.')
    return (losses.sum(1) / counts).mean()


def balanced_subset(rows, each=12):
    result=[]
    grouped=defaultdict(list)
    for row in rows:
        grouped[row['category']].append(row)
    for category in sorted(grouped):
        items = sorted(grouped[category], key=lambda r: hashlib.sha256(r['prompt'].encode()).hexdigest())
        result.extend(items[:each])
    return result


def evaluate_answers(model,tok,rows,device):
    model.eval()
    answers=[]
    for row in rows:
        reply,_=generate_chat_reply(model,tok,row['prompt'],max_new_tokens=100,greedy=True,device=device)
        answers.append(dict(row,reply=reply,exact=normalize(reply)==normalize(row['answer'])))
    scores={}
    for category in sorted(set(r['category'] for r in answers)):
        subset=[r for r in answers if r['category']==category]
        scores[category]={'correct':sum(r['exact'] for r in subset),'total':len(subset)}
    return {'scores':scores,'exact_rate':sum(r['exact'] for r in answers)/len(answers),'answers':answers}


@torch.no_grad()
def loss_on(model,data,device):
    model.eval()
    total,count=0.,0
    for x,y in DataLoader(data,batch_size=4,collate_fn=collate):
        x,y=x.to(device),y.to(device)
        with torch.amp.autocast('cuda',enabled=device.type=='cuda'):
            logits,_=model(x,y)
            loss=example_loss(logits,y)
        total+=loss.item()*len(x)
        count+=len(x)
    return total/count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default=str(ROOT/'checkpoints_chat_turns_20260927/best_model.pt'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=1600)
    parser.add_argument('--learning-rate',type=float,default=0.00012)
    parser.add_argument('--eval-every',type=int,default=400)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--accum',type=int,default=4)
    parser.add_argument('--seed',type=int,default=28)
    args=parser.parse_args()
    if min(args.steps,args.eval_every,args.batch_size,args.accum)<1:
        raise ValueError('Training sizes must be positive.')
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to replace existing run: {output}')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('This bounded laptop experiment requires CUDA; refusing a slow CPU run.')
    model,tok,info=load_model_for_inference(args.checkpoint,device=device)
    tok._tokenize_word=lru_cache(maxsize=65536)(tok._tokenize_word)
    rows=make_curriculum()
    accepted=[]
    rejected=Counter()
    seen=set()
    for row in rows:
        key=normalize(row['prompt'])
        if key in seen:
            continue
        try:
            encode_example(row,tok,min(192,model.config.max_seq_len))
        except ValueError as e:
            rejected[str(e)]+=1
            continue
        seen.add(key)
        accepted.append(row)
    splits={s:[r for r in accepted if r['split']==s] for s in ['train','validation','test']}
    # Synthetic task groups never cross splits. Facts intentionally do, through
    # different questions; those scores measure paraphrases, not unseen knowledge.
    for cat in ['grammar','arithmetic','comparison','reading']:
        groups={s:{r['group'] for r in data if r['category']==cat} for s,data in splits.items()}
        assert not (groups['train'] & groups['validation'] or groups['train'] & groups['test'] or groups['validation'] & groups['test'])
    for split,data in splits.items():
        (output/f'{split}.json').write_text(json.dumps(data,indent=2)+'\n')
    train=Examples(splits['train'],tok,192)
    validation_rows=balanced_subset(splits['validation'],each=24)
    validation=Examples(validation_rows,tok,192)
    probes=balanced_subset(splits['validation'],each=6)
    counts=Counter(row['category'] for row in train.rows)
    weights=[CATEGORY_WEIGHTS[row['category']]/counts[row['category']] for row in train.rows]
    sampler=WeightedRandomSampler(weights,args.steps*args.batch_size*args.accum,replacement=True,generator=torch.Generator().manual_seed(args.seed))
    loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,collate_fn=collate,num_workers=0)
    config=Config(model=model.config)
    config.training.learning_rate=args.learning_rate
    config.training.batch_size=args.batch_size
    config.training.grad_accum_steps=args.accum
    config.training.max_epochs=1
    config.system.checkpoints_dir=str(output)
    fingerprint=hashlib.sha256(Path(info['tokenizer']).read_bytes()).hexdigest()
    shutil.copy2(info['tokenizer'],output/'tokenizer_bpe.json')
    manifest={'source_model':info,'args':vars(args),'examples':{s:dict(Counter(r['category'] for r in data)) for s,data in splits.items()},'rejected':dict(rejected),'loss':'mean of per-example answer-token losses; prompts and padding masked','sampling':CATEGORY_WEIGHTS,'split_note':'Grammar, reading, comparison, and arithmetic groups are disjoint in this curriculum. Explanation tests use new wording for trained facts. Some basic facts/arithmetic appeared in earlier checkpoints. This is not a broad intelligence benchmark.','data_source':'Authored synthetic examples in capability_curriculum.py; no external dataset added in this run.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared dataset: '+json.dumps(manifest['examples']),flush=True)
    print('Rejected examples: '+json.dumps(dict(rejected)),flush=True)
    baseline_loss=loss_on(model,validation,device)
    baseline_dev=evaluate_answers(model,tok,probes,device)
    (output/'baseline_dev.json').write_text(json.dumps(baseline_dev,indent=2)+'\n')
    print(f'Baseline validation loss: {baseline_loss:.4f}; exact probes: {baseline_dev["exact_rate"]:.1%}',flush=True)
    groups=[{'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':.01},
            {'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0.}]
    optimizer=torch.optim.AdamW(groups,lr=args.learning_rate,betas=(.9,.95))
    scaler=torch.amp.GradScaler('cuda')
    best_key=(float('-inf'),float('-inf'))
    best_step=0
    history=[]
    start=time.monotonic()
    loss_sum=0.
    targets_seen=0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for micro,(x,y) in enumerate(loader,1):
        step=(micro-1)//args.accum+1
        progress=(step-1)/max(1,args.steps-1)
        lr=args.learning_rate*min(1.,step/20.)*(.2+.8*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:
            group['lr']=lr
        x,y=x.to(device),y.to(device)
        targets_seen+=int((y!=0).sum().item())
        with torch.amp.autocast('cuda'):
            logits,_=model(x,y)
            loss=example_loss(logits,y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss; checkpoint not promoted.')
        scaler.scale(loss/args.accum).backward()
        loss_sum+=loss.item()/args.accum
        if micro % args.accum:
            continue
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if step%50==0:
            elapsed=time.monotonic()-start
            print(f'Update {step}/{args.steps} | mean loss {loss_sum/50:.4f} | elapsed {elapsed:.0f}s | ETA {(args.steps-step)*elapsed/step:.0f}s | GPU allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB',flush=True)
            loss_sum=0.
        if step%args.eval_every==0 or step==args.steps:
            val_loss=loss_on(model,validation,device)
            dev=evaluate_answers(model,tok,probes,device)
            history.append({'step':step,'validation_loss':val_loss,'dev_exact':dev['exact_rate'],'scores':dev['scores']})
            (output/f'dev_{step}.json').write_text(json.dumps(dev,indent=2)+'\n')
            # Prefer demonstrated task success, then validation loss; never use test.
            key=(dev['exact_rate'],-val_loss)
            state={'model_state_dict':model.state_dict(),'config_dict':asdict(config),'global_step':step,'val_loss':val_loss,'tokenizer_sha256':fingerprint,'source_checkpoint':info['checkpoint'],'checkpoint_kind':'inference_weights_only','answer_target_tokens_seen':targets_seen}
            if key>best_key:
                temporary=output/'best_model.pt.partial'
                torch.save(state,temporary)
                temporary.replace(output/'best_model.pt')
                best_key,best_step=key,step
            print('Validation: '+json.dumps(history[-1]),flush=True)
            model.train()
    resume=dict(state,optimizer_state_dict=optimizer.state_dict(),scaler_state_dict=scaler.state_dict(),checkpoint_kind='training_resume',torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state())
    torch.save(resume,output/'last_training.pt')
    del resume,optimizer
    best=torch.load(output/'best_model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(best['model_state_dict'])
    test_rows=balanced_subset(splits['test'],each=16)
    candidate_test=evaluate_answers(model,tok,test_rows,device)
    (output/'candidate_test.json').write_text(json.dumps(candidate_test,indent=2)+'\n')
    del model,best
    torch.cuda.empty_cache()
    original,tok2,_=load_model_for_inference(args.checkpoint,device=device)
    tok2._tokenize_word=lru_cache(maxsize=65536)(tok2._tokenize_word)
    baseline_test=evaluate_answers(original,tok2,test_rows,device)
    (output/'baseline_test.json').write_text(json.dumps(baseline_test,indent=2)+'\n')
    report=dict(manifest,baseline_validation_loss=baseline_loss,history=history,best_step=best_step,
                elapsed_seconds=round(time.monotonic()-start,2),answer_target_tokens_seen=targets_seen,
                test_before=baseline_test['scores'],test_after=candidate_test['scores'],
                exact_before=baseline_test['exact_rate'],exact_after=candidate_test['exact_rate'])
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE: '+json.dumps({k:report[k] for k in ['best_step','elapsed_seconds','test_before','test_after','exact_before','exact_after']}),flush=True)


if __name__=='__main__':
    main()
