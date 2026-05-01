import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

#columns that need encoding (categorical)
need_onehot=['MS SubClass','MS Zoning','Street','Alley','Land Contour','Lot Config','Neighborhood','Condition 1','Condition 2','Bldg Type','House Style','Roof Style','Roof Matl','Exterior 1st','Exterior 2nd','Mas Vnr Type','Foundation','Heating','Central Air','Garage Type','Misc Feature','Sale Type','Sale Condition']
ordinal=['Lot Shape','Utilities','Land Slope','Overall Qual','Overall Cond','Exter Qual','Exter Cond','Bsmt Qual','Bsmt Cond','Bsmt Exposure','BsmtFin Type 1','BsmtFin Type 2','Heating QC','Electrical','Kitchen Qual','Functional','Fireplace Qu','Garage Finish','Garage Qual','Garage Cond','Paved Drive','Pool QC','Fence']
need_label_encoding=['Lot Shape','Utilities','Land Slope','Exter Qual','Exter Cond','Bsmt Qual','Bsmt Cond','Bsmt Exposure','BsmtFin Type 1','BsmtFin Type 2','Heating QC','Electrical','Kitchen Qual','Functional','Fireplace Qu','Garage Finish','Garage Qual','Garage Cond','Paved Drive','Pool QC','Fence']

ordinal_encoding_labels={
    'Lot Shape': ['Reg','IR1','IR2','IR3'],
    'Utilities': ['AllPub','NoSewr','NoSeWa'],
    'Land Slope': ['Gtl','Mod','Sev'],
    'Exter Qual': ['Ex','Gd','TA','Fa'],
    'Exter Cond': ['Ex','Gd','TA','Fa','Po'],
    'Bsmt Qual': ['Ex','Gd','TA','Fa','Po','No Basement'],
    'Bsmt Cond': ['Ex','Gd','TA','Fa','Po','No Basement'],
    'Bsmt Exposure': ['Gd','Av','Mn','No','No Basement'],
    'BsmtFin Type 1': ['GLQ','ALQ','BLQ','Rec','LwQ','Unf','No Basement'],
    'BsmtFin Type 2': ['GLQ','ALQ','BLQ','Rec','LwQ','Unf','No Basement'],
    'Heating QC': ['Ex','Gd','TA','Fa','Po'],
    'Electrical':['SBrkr','FuseA','FuseF','FuseP','Mix'],
    'Kitchen Qual': ['Ex','Gd','TA','Fa','Po'],
    'Functional': ['Typ','Min1','Min2','Mod','Maj1','Maj2','Sev','Sal'],
    'Fireplace Qu': ['Ex','Gd','TA','Fa','Po','No Fireplace'],
    'Garage Finish': ['Fin','RFn','Unf','No Garage'],
    'Garage Qual': ['Ex','Gd','TA','Fa','Po','No Garage'],
    'Garage Cond': ['Ex','Gd','TA','Fa','Po','No Garage'],
    'Paved Drive': ['Y','P','N'],
    'Pool QC': ['Ex','Gd','TA','Fa','No Pool'],
    'Fence': ['GdPrv','MnPrv','GdWo','MnWw','No Fence']
    }

def housing_preprocessing(file,target="Numerical"):
    df=pd.read_csv(file)

    if target=="Percentile":
        #Percentiles
        p20=df['SalePrice'].quantile(.2)
        p40=df['SalePrice'].quantile(.4)
        p60=df['SalePrice'].quantile(.6)
        p80=df['SalePrice'].quantile(.8)
        df['Percentile Bins']=pd.qcut(df['SalePrice'],q=[0, 0.2, 0.4, 0.6, 0.8, 1],labels=['< $124,000','$124,000-$146,500','$146,500-$178,536','$178,536-$230,000','> $230,000'])
        df=df.drop(columns=['SalePrice'])

    elif target=="Categorical":
        bins=[0, 100000, 180000, 260000, 400000, np.inf]
        df['Price Bins']=pd.cut(df['SalePrice'],bins=bins,labels=['< $100k','$100k-$180k','$180k-$260k','$260k-$400k','> $400k'],ordered=True)
        df=df.drop(columns=['SalePrice'])

    #Missing Values; Basement
    basement_cat=['Bsmt Qual','Bsmt Cond','Bsmt Exposure','BsmtFin Type 1','BsmtFin Type 2']
    all_catna=df[basement_cat].isnull().all(axis=1)
    df.loc[all_catna, basement_cat] = df.loc[all_catna, basement_cat].fillna({
        'Bsmt Qual': 'No Basement',
        'Bsmt Cond': 'No Basement',
        'Bsmt Exposure': 'No Basement',
        'BsmtFin Type 1': 'No Basement',
        'BsmtFin Type 2': 'No Basement'
    })

    #Missing Values; Fireplace
    df.loc[(df['Fireplace Qu'].isnull()) & (df['Fireplaces']==0),'Fireplace Qu'] = df.loc[(df['Fireplace Qu'].isnull()) & (df['Fireplaces']==0),'Fireplace Qu'].fillna('No Fireplace')

    #Missing Values; Garage
    gar_cat=['Garage Type','Garage Finish','Garage Qual','Garage Cond']
    gar_col=['Garage Type','Garage Finish','Garage Qual','Garage Cond','Garage Yr Blt']
    gar_catna=df[gar_col].isnull().all(axis=1)
    df.loc[gar_catna, gar_col] = df.loc[gar_catna, gar_col].fillna({
        'Garage Type': 'No Garage',
        'Garage Finish': 'No Garage',
        'Garage Qual': 'No Garage',
        'Garage Cond': 'No Garage'
        })
    df.loc[gar_catna, 'Garage Yr Blt']=0 #filling with arbitrary value to indicate no garage

    #Missing Values; Masonry
    df.loc[(df['Mas Vnr Type'].isnull()) & (df['Mas Vnr Area']==0),'Mas Vnr Type'] = df.loc[(df['Mas Vnr Type'].isnull()) & (df['Mas Vnr Area']==0),'Mas Vnr Type'].fillna('No Masonry')

    #Missing Values; Pool
    df.loc[(df['Pool QC'].isnull()) & (df['Pool Area']==0),'Pool QC'] = df.loc[(df['Pool QC'].isnull()) & (df['Pool Area']==0),'Pool QC'].fillna('No Pool')

    #Missing Valyes; Alley/Fence/Misc Feature
    df['Alley']=df['Alley'].fillna('No Alley Access')
    df['Fence']=df['Fence'].fillna('No Fence')
    df['Misc Feature']=df['Misc Feature'].fillna('No Misc Feature')

    #Missing Values; Lot Frontage; filling missing values with Neighborhood Median
    df['Lot Frontage']=df.groupby('Neighborhood')['Lot Frontage'].transform(lambda x: x.fillna(x.median()))

    #remove identifying columns
    df.drop(columns=['Order','PID'], inplace=True)

    #removing missing values
    df=df.dropna()

    return df

def full_df_prepared(file):
    df=housing_preprocessing(file,'Categorical')

    #drop missing values
    df=df.dropna()

    #numerical columns only
    nums=df.drop(columns=need_onehot)
    nums=nums.drop(columns=ordinal)
    nums=nums.drop(columns=['Price Bins'])

    #ordinal labeling
    ord_cols=df[need_label_encoding]
    for col in ord_cols.columns:
        encoder=OrdinalEncoder(categories=[ordinal_encoding_labels[col]])
        ord_cols[col]=encoder.fit_transform(ord_cols[[col]])

    #One-Hot Encoding
    encoder2=OneHotEncoder(sparse_output=False)
    encoder2.fit(df[need_onehot])
    encoded_cols=encoder2.get_feature_names_out(need_onehot)
    OHencoded=pd.DataFrame(encoder2.transform(df[need_onehot]),columns=encoded_cols)

    #combine segments
    full_df=pd.concat([nums.reset_index(drop=True),OHencoded.reset_index(drop=True),ord_cols.reset_index(drop=True),df[['Overall Qual','Overall Cond','Price Bins']].reset_index(drop=True)],axis=1)

    return full_df