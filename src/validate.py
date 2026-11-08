from pathlib import Path
import sqlite3,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
 c=pd.read_csv(ROOT/'data/clean/customers.csv'); p=pd.read_csv(ROOT/'data/clean/products.csv'); o=pd.read_csv(ROOT/'data/clean/orders.csv')
 raw_o=pd.read_csv(ROOT/'data/raw/orders.csv',dtype=str,keep_default_na=False); raw_c=pd.read_csv(ROOT/'data/raw/customers.csv',dtype=str,keep_default_na=False)
 assert len(raw_o)==13084 and raw_o.duplicated().sum()==119 and (raw_o.status=='pending').sum()==656 and raw_o.order_id.nunique()==12309
 assert pd.to_numeric(raw_o.order_amount,errors='coerce').isna().sum()==120 and (pd.to_numeric(raw_o.delivery_minutes,errors='coerce')<0).sum()==106 and (raw_o.delivery_minutes=='').sum()==2541
 assert (raw_c.age=='').sum()==61 and ((pd.to_numeric(raw_c.age,errors='coerce')<10)|(pd.to_numeric(raw_c.age,errors='coerce')>100)).sum()==6 and raw_c.city.nunique()==24 and raw_c.loc[raw_c.gender!='','gender'].nunique()==10 and (raw_c.gender=='').sum()==30
 assert len(c)==1200 and len(p)==40 and len(o)==12309 and o.order_id.nunique()==12309 and not (o.status=='pending').any()
 assert (o.status=='completed').sum()==11110 and (o.status=='cancelled').sum()==1199
 assert o.order_amount.isna().sum()==0 and o.loc[o.status=='completed','delivery_minutes'].isna().sum()==0 and o.loc[o.status=='cancelled','delivery_minutes'].isna().sum()==1199
 assert round(o.loc[o.status=='completed','order_amount'].sum(),2)==2998668.80
 with sqlite3.connect(ROOT/'data/instamart.db') as con:
  tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}; assert {'customers','products','orders'}<=tables
 print('ALL CHECKPOINTS PASSED')
if __name__=='__main__':main()
