from pathlib import Path
import sqlite3,pandas as pd,json
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
def main():
 sql=(ROOT/'sql/queries.sql').read_text(); chunks=[]
 for block in sql.split(';'):
  lines=[ln for ln in block.splitlines() if not ln.strip().startswith('--')]
  q='\n'.join(lines).strip()
  if q:chunks.append(q)
 results=[]
 with sqlite3.connect(ROOT/'data/instamart.db') as con:
  for i,q in enumerate(chunks,1):
   df=pd.read_sql_query(q,con); df.to_csv(OUT/f'q{i}_result.csv',index=False); results.append({'query':f'Q{i}','rows':len(df),'columns':list(df.columns)})
  c=pd.read_sql_query("SELECT c.city,SUM(o.order_amount) AS sum_inr,AVG(o.order_amount) AS mean_inr,COUNT(*) AS order_count FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.city ORDER BY c.city",con)
  c.to_csv(OUT/'city_spend_summary.csv',index=False)
  p=pd.read_sql_query("SELECT p.category, c.city, ROUND(SUM(o.order_amount),2) revenue FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY p.category,c.city",con)
  (OUT/'sql_analysis_summary.json').write_text(json.dumps({'query_results':results,'q2_rows':len(p),'q2_total_revenue':round(float(p.revenue.sum()),2),'q3_rows':len(pd.read_csv(OUT/'q3_result.csv')),'q4_rows':len(pd.read_csv(OUT/'q4_result.csv')),'q5_rows':len(pd.read_csv(OUT/'q5_result.csv')),'q7_rows':len(pd.read_csv(OUT/'q7_result.csv')),'q7_last_cumulative':float(pd.read_csv(OUT/'q7_result.csv').iloc[-1,-1])},indent=2))
 print('SQL analysis saved for',len(chunks),'queries')
if __name__=='__main__':main()
