from pathlib import Path
import sqlite3, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw'; CLEAN=ROOT/'data/clean'; OUT=ROOT/'outputs'; DB=ROOT/'data/instamart.db'
CLEAN.mkdir(parents=True,exist_ok=True); OUT.mkdir(exist_ok=True)
def main():
 c=pd.read_csv(RAW/'customers.csv',keep_default_na=False); p=pd.read_csv(RAW/'products.csv',keep_default_na=False); o=pd.read_csv(RAW/'orders.csv',keep_default_na=False); t=pd.read_csv(RAW/'tickets.csv',keep_default_na=False)
 # Audit snapshot, including per-column missingness.
 audit=[]
 for name,df in [('customers',c),('products',p),('orders',o),('tickets',t)]:
  for col in df.columns:
   miss=int(df[col].astype(str).str.strip().eq('').sum())
   audit.append({'table':name,'rows':len(df),'columns':len(df.columns),'column':col,'dtype':str(df[col].dtype),'missing_count':miss,'missing_pct':round(miss/len(df)*100,2)})
 pd.DataFrame(audit).to_csv(OUT/'raw_missingness_audit.csv',index=False)
 # Customer cleaning.
 c['city']=c.city.astype(str).str.strip().str.title(); c['city']=c.city.replace({'Bangalore':'Bengaluru'})
 def gender(v):
  z=str(v).strip().lower()
  if z in {'male','m'}:return 'male'
  if z in {'female','f'}:return 'female'
  return 'unknown'
 c['gender']=c.gender.map(gender); c['age']=pd.to_numeric(c.age,errors='coerce'); c.loc[~c.age.between(10,100),'age']=np.nan
 c['signup_date']=pd.to_datetime(c.signup_date,errors='coerce')
 c['age']=c['age'].fillna(c.groupby('city').age.transform('median')).round().astype(int)
 # Product table
 p['price_inr']=pd.to_numeric(p.price_inr,errors='raise')
 # Order cleaning in the order specified in the brief.
 o['order_ts']=pd.to_datetime(o.order_ts,errors='coerce'); o=o.drop_duplicates().copy()
 o=o.sort_values('order_ts').drop_duplicates('order_id',keep='last').copy()
 o['quantity']=pd.to_numeric(o.quantity,errors='coerce'); o['discount_pct']=pd.to_numeric(o.discount_pct,errors='coerce').fillna(0)
 o['order_amount']=pd.to_numeric(o.order_amount,errors='coerce'); o['delivery_minutes']=pd.to_numeric(o.delivery_minutes,errors='coerce'); o.loc[o.delivery_minutes<0,'delivery_minutes']=np.nan
 o=o.merge(p[['product_id','price_inr','category']],on='product_id',how='left',validate='many_to_one')
 missing=o.order_amount.isna(); o.loc[missing,'order_amount']=(o.loc[missing,'price_inr']*o.loc[missing,'quantity']*(1-o.loc[missing,'discount_pct']/100)).round(2)
 completed=o.status.eq('completed'); med=o.loc[completed].groupby('category').delivery_minutes.transform('median')
 o.loc[completed,'delivery_minutes']=o.loc[completed,'delivery_minutes'].fillna(med)
 # Cancelled delivery time remains missing; persist only required columns.
 o=o[['order_id','customer_id','product_id','quantity','order_ts','status','discount_pct','order_amount','delivery_minutes']]
 c.to_csv(CLEAN/'customers.csv',index=False,date_format='%Y-%m-%d'); p.to_csv(CLEAN/'products.csv',index=False); o.to_csv(CLEAN/'orders.csv',index=False,date_format='%Y-%m-%d %H:%M:%S')
 with sqlite3.connect(DB) as con:
  c.assign(signup_date=c.signup_date.dt.strftime('%Y-%m-%d')).to_sql('customers',con,index=False,if_exists='replace')
  p.to_sql('products',con,index=False,if_exists='replace')
  o.assign(order_ts=o.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')).to_sql('orders',con,index=False,if_exists='replace')
 print('Part A: clean rows',len(c),len(p),len(o),'completed',int((o.status=='completed').sum()),'cancelled',int((o.status=='cancelled').sum()),'completed revenue',round(o.loc[o.status=='completed','order_amount'].sum(),2))
 # Core validation checkpoints.
 assert len(o)==12309 and o.order_id.nunique()==12309 and not o.status.eq('pending').any()
 assert int((o.status=='completed').sum())==11110 and int((o.status=='cancelled').sum())==1199
 assert round(o.loc[o.status=='completed','order_amount'].sum(),2)==2998668.80
 assert o.loc[o.status=='completed','delivery_minutes'].isna().sum()==0
 assert o.loc[o.status=='cancelled','delivery_minutes'].isna().sum()==1199
 assert c.age.isna().sum()==0 and c.city.nunique()==6
 print('ALL CHECKPOINTS PASSED')
if __name__=='__main__':main()
