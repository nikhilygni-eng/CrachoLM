"""Independent before/after probes, including tasks outside the curriculum."""
import argparse
import gc
import json
from functools import lru_cache
from pathlib import Path
import time
import torch
from src.inference import load_model_for_inference
from src.chat_response import generate_chat_reply
from src.generator import generate_text
from src.data_general import CrachoGeneralDataset
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parent
EXTRA=[
    'Explain gravity to a child.',
    'Why does a laptop need RAM?',
    'Python list vs tuple?',
    'Rewrite naturally: yesterday i go shop and buyed two apple.',
    'Ravi has a blue hat, but Sara has a red hat. What color is Sara\'s hat?',
    'If I have 9 pens and give away 4, how many are left?',
    'What is 17 + 24?',
    'I want to learn Python. What should I do first?',
    'Write two sentences about a rainy morning.',
    'Explain why ice melts in warm water.',
    'Tell me about Bengaluru.',
    'Which number is greater: 18 or 27?',
]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--baseline-file',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    original=json.loads(Path(args.baseline_file).read_text())
    prompts=[r['prompt'] for r in original['rows']]+EXTRA
    results=[]
    for label,path in [('baseline',original['model']['checkpoint']),('candidate',args.candidate)]:
        model,tok,info=load_model_for_inference(path,device='cuda')
        tok._tokenize_word=lru_cache(maxsize=65536)(tok._tokenize_word)
        rows=[]
        for prompt in prompts:
            started=time.monotonic()
            reply,_=generate_chat_reply(model,tok,prompt,max_new_tokens=100,greedy=True,device='cuda')
            item={'prompt':prompt,'reply':reply,'seconds':round(time.monotonic()-started,3)}
            rows.append(item)
            print(label+': '+json.dumps(item),flush=True)
        ids=tok.encode((ROOT/'data_general/fluency_stories/validation.txt').read_text(),add_special_tokens=False)
        dataset=CrachoGeneralDataset(ids,model.config.max_seq_len)
        indices=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(28))[:32].tolist()
        total,count=0.,0
        with torch.inference_mode():
            for x,y in DataLoader(Subset(dataset,indices),batch_size=4):
                with torch.amp.autocast('cuda'):
                    _,loss=model(x.to('cuda'),y.to('cuda'))
                n=int((y!=0).sum())
                total+=loss.item()*n
                count+=n
        story_samples=[]
        for prompt in ['The dog saw a ball and','Tom was hungry, so he']:
            text=generate_text(model,tok,prompt,max_new_tokens=64,greedy=True,device='cuda')
            story_samples.append({'prompt':prompt,'text':text})
        results.append({'label':label,'model':info,'answers':rows,'story_validation_loss':total/count,'story_target_tokens':count,'stories':story_samples})
        del model,tok,dataset
        gc.collect()
        torch.cuda.empty_cache()
    Path(args.output).write_text(json.dumps({'note':'First 18 prompts are baseline diagnostic probes; extra probes include out-of-curriculum tasks. Some diagnostic questions match training examples and are not held-out scores. Story check uses 32 seeded chunks, not the entire validation corpus.','results':results},indent=2)+'\n')
    print('Independent review saved: '+args.output,flush=True)


if __name__=='__main__':
    main()
