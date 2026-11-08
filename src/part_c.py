from pathlib import Path
import pandas as pd, re, json, time, random, sys
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True); sys.path.insert(0,str(ROOT)); import mock_llm
EMAIL=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'); PHONE=re.compile(r'\b[6-9]\d{9}\b'); ORDER=re.compile(r'ORD-\d{6}')
ZERO='You are a ticket classifier. CLASSIFY each message into exactly one category: billing, delivery, technical, fraud. Reply only JSON of form {"category": "...", "reason": "..."}. Do not add prose.'
FEW=ZERO+'\nEXAMPLES:\nText: Unauthorized transaction, I did not place it. Reply: {"category":"fraud","reason":"unauthorized transaction"}\nText: My app crashes on login. Reply: {"category":"technical","reason":"app crashes"}\nText: My order is late. Reply: {"category":"delivery","reason":"late order"}'
ALLOWED={'billing','delivery','technical','fraud'}
def redact(text):
 counts={'email':0,'phone':0,'order_id':0}
 text,n=EMAIL.subn('[EMAIL]',text); counts['email']=n
 text,n=PHONE.subn('[PHONE]',text); counts['phone']=n
 text,n=ORDER.subn('[ORDER_ID]',text); counts['order_id']=n
 assert not EMAIL.search(text) and not PHONE.search(text) and not ORDER.search(text)
 return text,counts
def run(prompt,budget=100000,run_name='fewshot'):
 tickets=pd.read_csv(ROOT/'data/raw/tickets.csv',keep_default_na=False); segments=pd.read_csv(OUT/'customer_segments.csv'); seg=segments.set_index('customer_id').to_dict('index')
 mock_llm.reset(); running=0; results=[]; retry_log=[]; redaction={'email':0,'phone':0,'order_id':0}; classified=[]
 for _,r in tickets.iterrows():
  text=str(r.text); tokens=mock_llm.est_tokens(text); status=''; category=''; used=0
  if not text.strip():status='skipped_empty'
  elif tokens>400:status='flagged_long'
  elif running>=budget:status='budget_exhausted'
  else:
   safe,counts=redact(text)
   for k,v in counts.items():redaction[k]+=v
   # User message must be exactly the redacted ticket text.
   assert safe==safe and not EMAIL.search(safe) and not PHONE.search(safe) and not ORDER.search(safe)
   messages=[{'role':'system','content':prompt},{'role':'user','content':safe}]
   for retry in range(4):
    try:
     response=mock_llm.chat(messages); break
    except mock_llm.TransientLLMError:
     if retry>=3: raise
     wait=.05*(2**retry)*random.Random(42+len(retry_log)).uniform(.9,1.15); retry_log.append({'ticket_id':r.ticket_id,'retry_number':retry,'wait_seconds':wait}); time.sleep(wait)
   used=response.usage.total_tokens; running+=used; reply=response.choices[0].message.content or ''; parsed=None
   try: parsed=json.loads(reply)
   except (ValueError,TypeError):pass
   if isinstance(parsed,dict) and parsed.get('category') in ALLOWED:category=parsed['category'];status='classified_json'
   elif parsed is None:
    words=re.findall(r'[a-z]+',reply.lower()); matches=set(words)&ALLOWED
    if len(matches)==1:category=next(iter(matches));status='classified_keyword'
    else:status='flagged_invalid_output'
   else:status='flagged_invalid_output'
   if status in {'classified_json','classified_keyword'}:classified.append((r.true_category,category))
  customer_segment=seg.get(r.customer_id,{})
  segment_name=customer_segment.get('segment_name',''); premium=bool(customer_segment.get('is_premium',False))
  if status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}:route='human_review';priority='NORMAL'
  elif category=='fraud':route='security_team';priority='HIGH'
  elif premium:route='premium_desk';priority='HIGH'
  else:route='support_queue';priority='NORMAL'
  results.append({'ticket_id':r.ticket_id,'customer_id':r.customer_id,'status':status,'category':category,'segment_name':segment_name,'priority':priority,'route':route,'tokens_used':used,'true_category':r.true_category})
 df=pd.DataFrame(results); df.to_csv(OUT/('triage_budget_demo.csv' if budget<100000 else 'triage_results.csv'),index=False,columns=['ticket_id','customer_id','status','category','segment_name','priority','route','tokens_used'])
 metrics={'run':run_name,'budget':budget,'total_tokens':running,'classified':len(classified),'tokens_per_classified':running/max(len(classified),1),'accuracy':sum(a==b for a,b in classified)/max(len(classified),1),'status_counts':df.status.value_counts().to_dict(),'redaction_counts':redaction,'redaction_total':sum(redaction.values()),'retry_count':len(retry_log),'retries':retry_log,'budget_exhausted':int((df.status=='budget_exhausted').sum())}
 if run_name=='zero-shot':
  cats=sorted(ALLOWED); metrics['confusion_table']={a:{b:sum(x==a and y==b for x,y in classified) for b in cats} for a in cats}
 return metrics

def main():
 zero=run(ZERO,100000,'zero-shot'); few=run(FEW,100000,'few-shot'); budget=run(ZERO,6000,'budget-demo')
 (OUT/'triage_metrics.json').write_text(json.dumps({'zero_shot':zero,'few_shot':few,'budget_demo':budget},indent=2))
 print('Part C complete:',few['status_counts'],'zero-shot accuracy',round(zero['accuracy'],3),'few-shot accuracy',round(few['accuracy'],3),'budget exhausted',budget['budget_exhausted'])
if __name__=='__main__':main()
