from pathlib import Path
import sys, json, re, time, sqlite3, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; STATE=ROOT/'state'; OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True); sys.path.insert(0,str(ROOT)); import mock_llm
POL=ROOT/'data/policies'
def load_chunks(overlap=10):
 chunks=[]
 for path in sorted(POL.glob('*.txt')):
  words=path.read_text(encoding='utf-8').split(); start=0
  while start<len(words):
   part=words[start:start+40]; chunks.append({'source':path.name,'text':' '.join(part)})
   if start+40>=len(words):break
   start+=40-overlap
 return chunks
def retriever(overlap=10):
 chunks=load_chunks(overlap); vec=TfidfVectorizer(); mat=vec.fit_transform([c['text'] for c in chunks])
 def retrieve(query,k=3):
  scores=cosine_similarity(vec.transform([query]),mat).ravel(); idx=np.argsort(-scores,kind='stable')[:k]
  return [{'source':chunks[i]['source'],'score':float(scores[i]),'text':chunks[i]['text']} for i in idx]
 return retrieve,chunks
def answer(question,retrieve):
 docs=retrieve(question,3); context='\n'.join(f"[{d['source']}] {d['text']}" for d in docs)
 messages=[{'role':'system','content':'POLICY ASSISTANT: Answer only from the supplied context. If the answer is not in context, say you could not find it in the policy documents.'},{'role':'user','content':f'CONTEXT:\n{context}\nQUESTION:\n{question}'}]
 res=mock_llm.chat(messages); return res.choices[0].message.content,[d['source'] for d in docs],res.usage.total_tokens
TOOLS_SPEC=[{'name':'get_order','description':'Look up a single order in the SQLite database.','parameters':{'order_id':'string'}},{'name':'search_policy','description':'Retrieve policy chunks relevant to a query.','parameters':{'query':'string'}},{'name':'calculate_refund','description':'Calculate refund amount from order value and delivery time.','parameters':{'order_amount':'number','delivery_minutes':'number'}},{'name':'issue_refund','description':'Append an automatically approved refund to the ledger.','parameters':{'order_id':'string','refund_inr':'number'}},{'name':'escalate_to_human','description':'Record a refund requiring human approval.','parameters':{'order_id':'string','refund_inr':'number'}}]
def build_tools(retrieve):
 def get_order(order_id):
  with sqlite3.connect(ROOT/'data/instamart.db') as con:r=con.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone()
  return {'error':'order not found'} if not r else {'order_id':r[0],'status':r[1],'order_amount':float(r[2]),'delivery_minutes':None if pd.isna(r[3]) else float(r[3])}
 def search_policy(query):return {'results':retrieve(query,3)}
 def calculate_refund(order_amount,delivery_minutes):return {'refund_inr':round(float(order_amount),2) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0}
 def ledger(name,order_id,refund_inr):
  row={'timestamp':pd.Timestamp.now('UTC').isoformat(),'tool':name,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
  with open(OUT/'refund_ledger.jsonl','a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
  return {'refund_inr':row['refund_inr']}
 def issue_refund(order_id,refund_inr):return ledger('issue_refund',order_id,refund_inr)
 def escalate_to_human(order_id,refund_inr):return ledger('escalate_to_human',order_id,refund_inr)
 return {'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}
def run_agent(user_message,scenario_id,retrieve,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after=None,resume=False):
 tools=build_tools(retrieve); start=time.monotonic(); messages=[{'role':'system','content':'AGENT: process refund requests using tools and policy.'},{'role':'user','content':user_message}]; steps=[]; tokens=0; calls=0; run_id=run_id or f'{scenario_id}_{int(time.time()*1000000)}'; checkpoint=STATE/f'{run_id}.json'
 if resume:
  saved=json.loads(checkpoint.read_text()); messages=saved['messages']; steps=saved['steps']; tokens=saved['tokens_used']; calls=saved['llm_calls']
 while calls<max_iterations:
  if time.monotonic()-start>=time_limit_seconds:return {'status':'FALLBACK_TIMEOUT','final_answer':'FALLBACK_TIMEOUT: handing this case to a human agent.','steps':steps,'llm_calls':calls,'tokens_used':tokens,'run_id':run_id}
  response=mock_llm.chat(messages); calls+=1; tokens+=response.usage.total_tokens
  if tokens>token_limit:return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':'FALLBACK_TOKEN_LIMIT: handing this case to a human agent.','steps':steps,'llm_calls':calls,'tokens_used':tokens,'run_id':run_id}
  msg=response.choices[0].message
  if not msg.tool_calls:
   content=msg.content or ''; return {'status':content.split(':',1)[0],'final_answer':content,'steps':steps,'llm_calls':calls,'tokens_used':tokens,'run_id':run_id}
  tc=msg.tool_calls[0]; args=tc.arguments; name=tc.name
  messages.append({'role':'assistant','content':None,'tool_calls':[{'id':tc.id,'name':name,'arguments':args}]})
  observation=tools[name](**args); steps.append({'tool':name,'arguments':args,'observation':observation})
  messages.append({'role':'tool','tool_call_id':tc.id,'name':name,'content':json.dumps(observation)})
  checkpoint.write_text(json.dumps({'run_id':run_id,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens,'step_number':len(steps),'tools_executed':[s['tool'] for s in steps],'steps':steps,'llm_calls':calls},indent=2))
  if crash_after and len(steps)==crash_after:raise RuntimeError('SIMULATED_CRASH_AFTER_CHECKPOINT')
 return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':'FALLBACK_MAX_ITERATIONS: handing this case to a human agent with a summary.','steps':steps,'llm_calls':calls,'tokens_used':tokens,'run_id':run_id}
def resume_agent(run_id,retrieve):return run_agent('',run_id.split('_')[0],retrieve,run_id=run_id,resume=True)
def main():
 retrieve,chunks=retriever(10); assert len(chunks)==15, f'expected 15 chunks, got {len(chunks)}'
 evaldf=pd.read_csv(ROOT/'data/raw/policy_eval.csv',keep_default_na=False); eval_rows=[]
 for _,r in evaldf.iterrows():
  ans,src,toks=answer(r.question,retrieve); eval_rows.append({'question':r.question,'expected_doc':r.expected_doc,'retrieved_sources':';'.join(src),'expected_doc_in_top3':r.expected_doc in src if r.expected_doc!='NONE' else False,'expected_fact_found':r.expected_fact in ans if r.expected_fact and r.expected_doc!='NONE' else False,'answer':ans})
 pd.DataFrame(eval_rows).to_csv(OUT/'policy_eval_results.csv',index=False)
 retrieve0,chunks0=retriever(0); no_rows=[]
 for _,r in evaldf.iterrows():
  ans,src,_=answer(r.question,retrieve0); no_rows.append({'question':r.question,'expected_doc':r.expected_doc,'expected_doc_in_top3':r.expected_doc in src if r.expected_doc!='NONE' else False,'expected_fact_found':r.expected_fact in ans if r.expected_fact and r.expected_doc!='NONE' else False,'answer':ans})
 pd.DataFrame(no_rows).to_csv(OUT/'policy_eval_zero_overlap.csv',index=False)
 # Demonstration retrieval API output
 (OUT/'policy_retrieval_example.json').write_text(json.dumps(retrieve('refund policy late delivery',3),indent=2))
 # Agent scenarios and ledger.
 (OUT/'refund_ledger.jsonl').write_text('')
 scenarios=pd.read_csv(ROOT/'data/raw/agent_scenarios.csv',keep_default_na=False); runs=[]
 for _,r in scenarios.iterrows():
  result=run_agent(r.message,r.scenario_id,retrieve,max_iterations=6,token_limit=6000,time_limit_seconds=10)
  row={'scenario_id':r.scenario_id,**{k:result[k] for k in ['status','final_answer','llm_calls','tokens_used','steps']},'expected_outcome':r.expected_outcome,'expected_refund_inr':float(r.expected_refund_inr)}; runs.append(row)
 with open(OUT/'agent_runs.jsonl','w',encoding='utf-8') as f:
  for row in runs:f.write(json.dumps(row)+'\n')
 # Stop-safely demos.
 s1=scenarios.iloc[0]; low=run_agent(s1.message,'S1',retrieve,token_limit=200); timeout=run_agent(s1.message,'S1',retrieve,time_limit_seconds=0)
 # Crash after second tool execution, then continue the exact checkpoint (no tool replay).
 crashid='S1_crash_resume';
 try:run_agent(s1.message,'S1',retrieve,run_id=crashid,crash_after=2)
 except RuntimeError:pass
 resumed=resume_agent(crashid,retrieve)
 demo={'token_limit_demo':{'status':low['status'],'llm_calls':low['llm_calls']},'timeout_demo':{'status':timeout['status'],'llm_calls':timeout['llm_calls']},'crash_resume':{'status':resumed['status'],'final_answer':resumed['final_answer'],'tools_executed':[s['tool'] for s in resumed['steps']],'llm_calls':resumed['llm_calls']}}
 (OUT/'agent_safety_demos.json').write_text(json.dumps(demo,indent=2))
 print('Part D complete: chunks',len(chunks),'scenarios',[(r['scenario_id'],r['status'],r['llm_calls']) for r in runs]); print('Demos',demo)
if __name__=='__main__':main()
