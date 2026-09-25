
#!/usr/bin/env python3
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
CLASS_NAMES = {
"1_banking_kyc_otp_fraud":"Banking/KYC/OTP","2_upi_wallet_fraud":"UPI/Wallet",
"3_investment_task_scam":"Investment/Task","4_digital_arrest_govt_impersonation":"Digital Arrest/Govt",
"5_loan_credit_app_scams":"Loan/Credit","6_delivery_customer_care_scams":"Delivery/Customer Care",
"7_dating_romance_sextortion":"Dating/Romance/Sextortion","8_legacy_telecom_scams":"Legacy Telecom"}

def words(text): return TOKEN_RE.findall(text.lower())
def load(root):
    out=[]
    for d in sorted(Path(root).iterdir()):
        if d.is_dir() and d.name in CLASS_NAMES:
            for p in sorted(d.glob('*.txt')):
                out.append({'id':p.stem,'class_id':d.name,'class_name':CLASS_NAMES[d.name],
                            'path':str(p),'text':p.read_text(encoding='utf-8',errors='ignore')})
    return out

def pct(a,b): return 100*a/b if b else 0.0
def mean(xs): return float(np.mean(xs)) if xs else 0.0
def med(xs): return float(np.median(xs)) if xs else 0.0

def analyze(tok, vocab, rec):
    text=rec['text']; ws=words(text); nwords=len(ws)
    enc=tok(text,add_special_tokens=True,truncation=False,padding=False,return_attention_mask=False)
    ids=[int(x) for x in enc['input_ids']]; special=set(tok.all_special_ids)
    content=[x for x in ids if x not in special]
    unk=set([int(tok.unk_token_id)]) if tok.unk_token_id is not None else set()
    unk_n=sum(x in unk for x in content)
    counts=[]
    try:
        e=tok(text,add_special_tokens=False,truncation=False,padding=False,return_offsets_mapping=True)
        spans=[(m.start(),m.end()) for m in TOKEN_RE.finditer(text.lower())]
        for a,b in spans:
            c=sum(1 for sid,(x,y) in zip(e['input_ids'],e['offset_mapping']) if int(sid) not in special and y>a and x<b)
            counts.append(c)
        if len(counts)!=len(spans): raise ValueError
    except Exception:
        counts=[]
        for w in ws:
            x=tok(w,add_special_tokens=False,truncation=False,padding=False)['input_ids']
            counts.append(sum(int(i) not in special for i in x))
    return {'id':rec['id'],'class_id':rec['class_id'],'class_name':rec['class_name'],
      'n_words':nwords,'encoder_tokens_raw':len(content),'encoder_tokens_with_special':len(ids),
      'word_oov_count':sum(w not in vocab for w in ws),'word_oov_rate':pct(sum(w not in vocab for w in ws),nwords),
      'unk_count':unk_n,'unk_rate':pct(unk_n,len(content)),
      'mean_subwords_per_word':mean(counts),'median_subwords_per_word':med(counts),
      'split_ge_2_rate':pct(sum(c>=2 for c in counts),nwords),'split_ge_3_rate':pct(sum(c>=3 for c in counts),nwords),
      'split_ge_4_rate':pct(sum(c>=4 for c in counts),nwords),'expansion_ratio':len(content)/nwords if nwords else 0,
      'over_512':int(len(ids)>512),'over_1024':int(len(ids)>1024),'over_2048':int(len(ids)>2048),
      'over_4096':int(len(ids)>4096),'over_8192':int(len(ids)>8192)}

def aggregate(rs):
    if not rs:return {}
    def m(k):return mean([float(r[k]) for r in rs])
    return {'conversations':len(rs),'mean_words':m('n_words'),'median_words':med([r['n_words'] for r in rs]),
      'mean_encoder_tokens_raw':m('encoder_tokens_raw'),'median_encoder_tokens_raw':med([r['encoder_tokens_raw'] for r in rs]),
      'p95_encoder_tokens_raw':float(np.percentile([r['encoder_tokens_raw'] for r in rs],95)),
      'word_oov_rate_mean_per_conversation':m('word_oov_rate'),'unk_rate_mean_per_conversation':m('unk_rate'),
      'mean_subwords_per_word':m('mean_subwords_per_word'),'median_subwords_per_word':med([r['median_subwords_per_word'] for r in rs]),
      'split_ge_2_rate_mean':m('split_ge_2_rate'),'split_ge_3_rate_mean':m('split_ge_3_rate'),'split_ge_4_rate_mean':m('split_ge_4_rate'),
      'expansion_ratio_mean':m('expansion_ratio'),
      'pct_conversations_over_512':pct(sum(r['over_512'] for r in rs),len(rs)),
      'pct_conversations_over_1024':pct(sum(r['over_1024'] for r in rs),len(rs)),
      'pct_conversations_over_2048':pct(sum(r['over_2048'] for r in rs),len(rs)),
      'pct_conversations_over_4096':pct(sum(r['over_4096'] for r in rs),len(rs)),
      'pct_conversations_over_8192':pct(sum(r['over_8192'] for r in rs),len(rs))}

def write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description='Subword/BPE analysis for RoBERTa and ModernBERT')
    ap.add_argument('--train',required=True);ap.add_argument('--test',required=True);ap.add_argument('--yt',required=True)
    ap.add_argument('--models',nargs='+',default=['FacebookAI/roberta-base','answerdotai/ModernBERT-base'])
    ap.add_argument('--output',default='analysis/bpe_tokenizer');args=ap.parse_args()
    tr,te,yt=load(args.train),load(args.test),load(args.yt); vocab=set(w for r in tr for w in words(r['text']))
    summary=[]; classes=[]; conv=[]
    for model in args.models:
        name=model.split('/')[-1]; print('\nLoading',model); tok=AutoTokenizer.from_pretrained(model,use_fast=True)
        for split,recs in [('train',tr),('test',te),('yt',yt)]:
            rs=[]
            for i,r in enumerate(recs,1):
                a=analyze(tok,vocab,r);a['model']=name;a['split']=split;rs.append(a);conv.append(a)
                if i%250==0:print(split,i)
            s=aggregate(rs);s.update(model=name,split=split);summary.append(s)
            by=defaultdict(list)
            for r in rs:by[r['class_name']].append(r)
            for cls,g in sorted(by.items()):
                a=aggregate(g);a.update(model=name,split=split,class_name=cls);classes.append(a)
            fields=['id','class_id','class_name','n_words','encoder_tokens_raw','encoder_tokens_with_special','word_oov_count','word_oov_rate','unk_count','unk_rate','mean_subwords_per_word','median_subwords_per_word','split_ge_2_rate','split_ge_3_rate','split_ge_4_rate','expansion_ratio','over_512','over_1024','over_2048','over_4096','over_8192','model','split']
            write(Path(args.output)/f'{name}_{split}_per_conversation.csv',[{k:r[k] for k in fields} for r in rs])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    write(out/'bpe_summary.csv',summary);write(out/'bpe_by_class.csv',classes);write(out/'bpe_per_conversation.csv',conv)
    deltas=[]
    for model in sorted({r['model'] for r in summary}):
        q={r['split']:r for r in summary if r['model']==model}
        if 'test' in q and 'yt' in q:
            for k in ['mean_subwords_per_word','split_ge_2_rate_mean','split_ge_3_rate_mean','split_ge_4_rate_mean','expansion_ratio_mean','mean_encoder_tokens_raw','p95_encoder_tokens_raw','word_oov_rate_mean_per_conversation','unk_rate_mean_per_conversation','pct_conversations_over_512','pct_conversations_over_1024','pct_conversations_over_2048','pct_conversations_over_4096','pct_conversations_over_8192']:
                deltas.append({'model':model,'metric':k,'test_value':q['test'][k],'yt_value':q['yt'][k],'yt_minus_test':q['yt'][k]-q['test'][k]})
    write(out/'yt_vs_test_delta.csv',deltas)
    with (out/'bpe_summary.json').open('w') as f:json.dump({'train_word_vocabulary_size':len(vocab),'summary':summary,'by_class':classes},f,indent=2)
    print('\nDone. Results:',out)
if __name__=='__main__':main()
