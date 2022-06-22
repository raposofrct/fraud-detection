import pandas as pd
import numpy as np
import inflection
import pickle as pkl

class Fraud():
    def __init__(self):
        self.le = pkl.load(open('pkl/le.pkl','rb'))
        self.ohe = pkl.load(open('pkl/ohe.pkl','rb'))
        self.mms = pkl.load(open('pkl/mms.pkl','rb'))
        
    def data_cleaning(self,dados):
        
        ## 2.1 Rename Values and Columns

        for column in dados.columns:
            dados.rename(columns={column:inflection.underscore(column)},inplace=True)

        dados['type'] = dados['type'].apply(lambda x: inflection.underscore(x))

        dados.rename(columns={'oldbalance_org':'old_balance_orig','newbalance_orig':'new_balance_orig','oldbalance_dest':'old_balance_dest','newbalance_dest':'new_balance_dest'},inplace=True)
        
        # podia usar um dados.fillna('algum valor ou média dos valores usando simple imputer')
        # pq se aparecer um dado faltante eu ns oq fazer e dará erro! (melhorar nisso talvez!)

        return dados
    
    def feature_engineering(self,dados):

        # Type of destination client
        dados['type_dest'] = dados['name_dest'].astype(str).str[0].map({'C':'customers','M':'merchants'})

        # If the origin account has lost or earned money or if nothing has changed in that transaction
        dados['orig_transaction_status'] = dados.apply(lambda x: 'orig_lost_money' if x['new_balance_orig']<x['old_balance_orig'] else ('no_change' if x['new_balance_orig']==x['old_balance_orig'] else 'orig_earned_money'),1)

        # If the destination account has lost or earned money or if nothing has changed in that transaction
        dados['dest_transaction_status'] = dados.apply(lambda x: 'dest_lost_money' if x['new_balance_dest']<x['old_balance_dest'] else ('no_change' if x['new_balance_dest']==x['old_balance_dest'] else 'dest_earned_money'),1)

        # Orig account has no money after the transaction
        dados['orig_has_no_money'] = dados['new_balance_orig'].apply(lambda x: 1 if x==0 else 0)

        # Dest account had no money before the transaction
        dados['dest_had_no_money'] = dados['old_balance_dest'].apply(lambda x: 1 if x==0 else 0)
        
        return dados
        
    def data_filtering(self,dados):

        dados.drop(['name_orig','name_dest'],1,inplace=True) # preciso da feature dest pra o feature engineering, a outra eu poderia nem filtrar, só não usar como input! (melhorar depois)
        
        return dados
    
    def data_preparation(self,dados):

        ## 6.2 Encoding

        # type (multiclass)
        dados = self.ohe.transform(dados).drop('type_debit',1)
        dados['type_dest'] = self.le.transform(dados['type_dest'])
        dados['dest_transaction_status'] = dados['dest_transaction_status'].map({'dest_earned_money':1, 'dest_lost_money':-1, 'no_change':0})
        dados['orig_transaction_status'] = dados['orig_transaction_status'].map({'orig_earned_money':1, 'orig_lost_money':-1, 'no_change':0})

        ## 6.4 Rescale

        columns = ['step', 'type_transfer', 'type_cash_out', 'type_payment','type_cash_in', 'amount', 'old_balance_orig', 'new_balance_orig','old_balance_dest', 'new_balance_dest', 'is_flagged_fraud', 'type_dest','orig_transaction_status', 'dest_transaction_status','orig_has_no_money', 'dest_had_no_money']
        dados = pd.DataFrame(self.mms.transform(dados),columns=columns)

        ## 6.5 Feature Selection

        cols_selected = ['step','old_balance_orig','new_balance_dest','dest_transaction_status','orig_transaction_status','amount','type_payment','type_transfer','type_dest','orig_has_no_money']
        dados = dados[cols_selected]
        
        return dados
    
    def get_predictions(self,model,dados):
        
        y_pred = pd.Series(model.predict_proba(x_test)[:,1],name='proba').apply(lambda x: 1 if x>= 0.4 else 0)
        return y_pred.to_json(orient='records')