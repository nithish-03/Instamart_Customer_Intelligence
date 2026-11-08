from pathlib import Path
import pandas as pd, numpy as np, joblib, json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
def main():
 orders=pd.read_csv(ROOT/'data/clean/orders.csv',parse_dates=['order_ts']); customers=pd.read_csv(ROOT/'data/clean/customers.csv',parse_dates=['signup_date']); products=pd.read_csv(ROOT/'data/clean/products.csv')
 done=orders[orders.status.eq('completed')].copy(); snap=pd.Timestamp('2025-10-01')
 rfm=done.groupby('customer_id').agg(last_order_ts=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index(); rfm['recency']=(snap-rfm.last_order_ts).dt.days; rfm=rfm[['customer_id','recency','frequency','monetary']]
 scaler=StandardScaler(); X=scaler.fit_transform(rfm[['recency','frequency','monetary']]); krows=[]
 for k in range(2,9):
  km=KMeans(n_clusters=k,n_init=10,random_state=42); labels=km.fit_predict(X); krows.append({'k':k,'inertia':km.inertia_,'silhouette':silhouette_score(X,labels)})
 pd.DataFrame(krows).to_csv(OUT/'kmeans_metrics.csv',index=False)
 # K=4 follows the brief's reference profile and provides actionable segments.
 km=KMeans(n_clusters=4,n_init=10,random_state=42); rfm['segment_id']=km.fit_predict(X)
 profile=rfm.groupby('segment_id').agg(size=('customer_id','size'),mean_recency=('recency','mean'),mean_frequency=('frequency','mean'),mean_monetary=('monetary','mean')).sort_values('mean_monetary',ascending=False)
 names={}; premium_id=int(profile.index[0]);
 for i,row in profile.iterrows():
  if i==premium_id:names[i]='Premium Champions'
  elif row.mean_frequency>=rfm.frequency.mean():names[i]='Frequent Essentials'
  elif row.mean_recency>rfm.recency.median():names[i]='At-Risk Customers'
  else:names[i]='Occasional Shoppers'
 rfm['segment_name']=rfm.segment_id.map(names); rfm['is_premium']=rfm.segment_id.eq(premium_id)
 rfm[['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium']].to_csv(OUT/'customer_segments.csv',index=False)
 pca=PCA(n_components=2).fit(X); (OUT/'pca_variance.json').write_text(json.dumps({'explained_variance_ratio':pca.explained_variance_ratio_.tolist()},indent=2))
 # Lapse cohort and pre-cutoff features only.
 cutoff=pd.Timestamp('2025-09-01'); before=orders[(orders.order_ts<cutoff)].copy(); bdone=before[before.status.eq('completed')]; cohort=bdone.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean')).reset_index(); cohort['recency']=(cutoff-cohort.last_order).dt.days
 final=before[before.status.isin(['completed','cancelled'])].groupby('customer_id').status.agg(total='count',cancelled=lambda s:(s=='cancelled').sum()).reset_index(); cohort=cohort.merge(final,on='customer_id',how='left'); cohort['cancellation_rate']=cohort.cancelled.div(cohort.total).fillna(0)
 fav=bdone.merge(products[['product_id','category']],on='product_id').groupby('customer_id').category.agg(lambda s:s.mode().iloc[0]).rename('favourite_category').reset_index(); cohort=cohort.merge(fav,on='customer_id',how='left').merge(customers,on='customer_id',how='left')
 cohort['tenure_days']=(cutoff-cohort.signup_date).dt.days
 september=orders[(orders.order_ts>=cutoff)&(orders.order_ts<pd.Timestamp('2025-10-01'))&orders.status.eq('completed')].groupby('customer_id').order_id.nunique()
 cohort['lapsed']=(~cohort.customer_id.isin(september.index)).astype(int); cohort=cohort.drop(columns=['last_order','total','cancelled']); cohort=cohort.dropna(subset=['favourite_category','city','gender','age','tenure_days'])
 features=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','favourite_category','city','age','gender','tenure_days']; Xdf=cohort[features]; y=cohort.lapsed.astype(int)
 X_train,X_test,y_train,y_test=train_test_split(Xdf,y,test_size=.2,stratify=y,random_state=42)
 num=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','age','tenure_days']; cat=['favourite_category','city','gender']
 pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(drop='first',handle_unknown='ignore'))]),cat)])
 lr=Pipeline([('prep',pre),('model',LogisticRegression(max_iter=1000))]); gs_lr=GridSearchCV(lr,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1); gs_lr.fit(X_train,y_train)
 rf=Pipeline([('prep',pre),('model',RandomForestClassifier(n_estimators=100,random_state=42))]); gs_rf=GridSearchCV(rf,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1); gs_rf.fit(X_train,y_train)
 def metrics(model):
  pred=model.predict(X_test); return {'best_params':getattr(model,'best_params_',{}),'cv_f1':float(model.best_score_),'confusion_matrix':confusion_matrix(y_test,pred).tolist(),'precision':precision_score(y_test,pred,zero_division=0),'recall':recall_score(y_test,pred,zero_division=0),'f1':f1_score(y_test,pred,zero_division=0),'accuracy':accuracy_score(y_test,pred)}
 # Prefer RF if it has higher CV F1; persist a complete sklearn pipeline.
 chosen=gs_rf if gs_rf.best_score_>=gs_lr.best_score_ else gs_lr; joblib.dump(chosen.best_estimator_,OUT/'lapse_model.joblib'); loaded=joblib.load(OUT/'lapse_model.joblib'); assert np.array_equal(chosen.best_estimator_.predict(X_test),loaded.predict(X_test))
 pred=loaded.predict(X_test); probs=loaded.predict_proba(X_test)[:,1]; res={'rf':metrics(gs_rf),'logistic':metrics(gs_lr),'selected':'random_forest' if chosen is gs_rf else 'logistic_regression','test_rows':len(y_test),'cohort_rows':len(y),'lapsed_count':int(y.sum()),'baseline_accuracy':float(max(y_test.mean(),1-y_test.mean())),'selected_f1':float(f1_score(y_test,pred,zero_division=0)),'selected_model_reloaded_predictions_match':True}
 (OUT/'lapse_metrics.json').write_text(json.dumps(res,indent=2))
 testids=cohort.loc[X_test.index,['customer_id','lapsed']].copy(); testids['lapse_probability']=probs; seg=pd.read_csv(OUT/'customer_segments.csv'); testids=testids.merge(seg[['customer_id','is_premium']],on='customer_id',how='left'); top=testids[testids.is_premium.fillna(False)].sort_values('lapse_probability',ascending=False).head(10); top.rename(columns={'lapsed':'actually_lapsed'})[['customer_id','lapse_probability','actually_lapsed']].to_csv(OUT/'retention_list.csv',index=False)
 # Simple training-only oversampling experiment, applied to the best logistic pipeline.
 from sklearn.utils import resample
 train=pd.DataFrame({'idx':range(len(y_train)),'y':y_train.to_numpy()}); maj=train[train.y==train.y.value_counts().idxmax()]; minc=train[train.y!=train.y.value_counts().idxmax()]; add=resample(minc,replace=True,n_samples=len(maj)-len(minc),random_state=42); balanced=pd.concat([train,add]); model_bal=gs_lr.best_estimator_; model_bal.fit(X_train.iloc[balanced.idx.to_numpy()],y_train.iloc[balanced.idx.to_numpy()]); bp=model_bal.predict(X_test); res['balanced_logistic_test']={'precision':precision_score(y_test,bp,zero_division=0),'recall':recall_score(y_test,bp,zero_division=0),'f1':f1_score(y_test,bp,zero_division=0)}; (OUT/'lapse_metrics.json').write_text(json.dumps(res,indent=2))
 # Interpretability artifacts for the README.
 try:
  prep=gs_rf.best_estimator_.named_steps['prep']; model=gs_rf.best_estimator_.named_steps['model']; names_out=prep.get_feature_names_out()
  importance=sorted([{'feature':str(n),'importance':float(v)} for n,v in zip(names_out,model.feature_importances_)],key=lambda z:z['importance'],reverse=True)[:5]
  lr_model=gs_lr.best_estimator_.named_steps['model']; lr_names=gs_lr.best_estimator_.named_steps['prep'].get_feature_names_out()
  coefs=sorted([{'feature':str(n),'coefficient':float(v),'sign':'positive' if v>0 else 'negative' if v<0 else 'zero'} for n,v in zip(lr_names,lr_model.coef_[0])],key=lambda z:abs(z['coefficient']),reverse=True)[:10]
  (OUT/'model_interpretation.json').write_text(json.dumps({'random_forest_top5':importance,'logistic_largest_coefficients':coefs},indent=2))
 except Exception as e: (OUT/'model_interpretation.json').write_text(json.dumps({'note':str(e)},indent=2))
 print('Part B complete: RFM customers',len(rfm),'cohort',len(cohort),'selected test F1',round(res['selected_f1'],3))
if __name__=='__main__':main()
