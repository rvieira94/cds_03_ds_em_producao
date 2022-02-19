import pickle
import pandas as pd
import inflection
import numpy as np
import datetime
import math

class Rossmann ( object ):
    
    def __init__( self ):
        self.home='/PATH/'
        self.competition_distance_scaler   = pickle.load(open(self.home + 'parameter/competition_distance_scaler','rb'))
        self.year_scaler                   = pickle.load(open(self.home + 'parameter/year_scaler','rb'))
        self.competition_time_month_scaler = pickle.load(open(self.home + 'parameter/competition_time_month_scaler','rb'))
        self.promo_time_week_scaler        = pickle.load(open(self.home + 'parameter/promo_time_week_scaler','rb'))
        self.store_type_encoding           = pickle.load(open(self.home + 'parameter/store_type_encoding','rb'))
        self.model_xgb                     = pickle.load(open(self.home + 'model/model_rossmann.pkl', 'rb') )

        
    def data_cleaning( self, df1 ):
        
        ## 1.1. Rename Columns
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday',
                    'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
        snake_case = lambda x: inflection.underscore(x)
        cols_new = list(map(snake_case,cols_old))
        df1.columns = cols_new

        ## 1.3. Data Types
        df1['date'] = pd.to_datetime(df1['date'])

        ## 1.5. Fillout NAs

        # competition_distance: 
        df1['competition_distance'] = df1['competition_distance'].apply( lambda x: 200000.0 if math.isnan(x) else x )

        # competition_open_since_month 
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1 )

        # competition_open_since_year
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1 )

        # promo2_since_week 
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1 )         

        # promo2_since_year
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1 )                 

        # promo_interval  
        month_map = { 1 : 'Jan', 2 : 'Feb', 3 : 'Mar', 4 : 'Apr', 5 : 'May', 6 : 'Jun', 7 : 'Jul', 8 : 'Aug', 9 : 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec' } 
        df1['promo_interval'].fillna(0, inplace=True)
        df1['month_map'] = df1['date'].dt.month.map( month_map ) # extrai o mês int e mapeia com o dicionario, tranformando em string
        df1['is_promo'] = df1[['promo_interval','month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        ## 1.6. Change Types

        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)
        
        return df1
    
    def feature_engineering( self, df3 ):
        # year
        df3['year'] = df3['date'].dt.year

        # month
        df3['month'] = df3['date'].dt.month

        # day
        df3['day'] = df3['date'].dt.day

        # week of year
        df3['week_of_year'] = df3['date'].dt.weekofyear

        # year week
        df3['year_week'] = df3['date'].dt.strftime('%Y-%W')

        # competition since
        df3['competition_since'] = df3.apply( lambda x: datetime.datetime( year = x['competition_open_since_year'],
                                                                          month = x['competition_open_since_month'], 
                                                                          day=1), axis=1 )
        df3['competition_time_month'] = ( ( df3['date'] - df3['competition_since'] ) / 30 ).apply( lambda x: x.days ).astype(int)

        # promo since
        df3['promo_since'] = df3['promo2_since_year'].astype(str) + '-' + df3['promo2_since_week'].astype(str)
        df3['promo_since'] = df3['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        df3['promo_time_week'] = ( ( df3['date'] - df3['promo_since'] ) / 7 ).apply( lambda x: x.days ).astype(int)

        # assortment
        df3['assortment'] = df3['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        # state holiday
        df3['state_holiday'] = df3['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        # 4.0. Variables filtering
        ## 4.1. Rows filtering
        df3 = df3[df3['open'] != 0]

        ## 4.2. Columns selection

        cols_drop = ['open','promo_interval','month_map']
        df3 = df3.drop(cols_drop, axis=1)
        
        return df3
    
    def data_preparation( self, df6 ):

        ## 6.2. Rescaling
        # competition_distance  
        df6['re_competition_distance'] = self.competition_distance_scaler.transform(df6[['competition_distance']].values)

        # year                  
        df6['re_year'] = self.year_scaler.transform(df6[['year']].values)

        # competition_time_month
        df6['re_competition_time_month'] = self.competition_time_month_scaler.transform(df6[['competition_time_month']].values)

        # promo_time_week       
        df6['re_promo_time_week'] = self.promo_time_week_scaler.transform(df6[['promo_time_week']].values)


        ## 6.3. Transformação
        ### 6.3.1. Encoding

        # state_holiday - One Hot Encoding
        df6 = pd.get_dummies(df6, prefix=['state_holiday'], columns=['state_holiday'])

        # store_type - Label Encoding
        df6['store_type'] = self.store_type_encoding.transform(df6['store_type'])

        # assortment - Ordinal Encoding
        assortment = { 'basic': 1,'extra': 2, 'extended': 3 }

        df6['assortment'] = df6['assortment'].map(assortment)

        ### 6.3.3. Nature Transformation (variaveis cíclicas)

        # day_of_week
        df6['day_of_week_sin'] = df6['day_of_week'].apply(lambda x: np.sin( x * ( 2 * np.pi / 7 ) ) )
        df6['day_of_week_cos'] = df6['day_of_week'].apply(lambda x: np.cos( x * ( 2 * np.pi / 7 ) ) )

        # day          
        df6['day_sin'] = df6['day'].apply(lambda x: np.sin( x * ( 2 * np.pi / 30 ) ) )
        df6['day_cos'] = df6['day'].apply(lambda x: np.cos( x * ( 2 * np.pi / 30 ) ) )

        # month
        df6['month_sin'] = df6['month'].apply(lambda x: np.sin( x * ( 2 * np.pi / 12 ) ) )
        df6['month_cos'] = df6['month'].apply(lambda x: np.cos( x * ( 2 * np.pi / 12 ) ) )

        # week_of_year 
        df6['week_of_year_sin'] = df6['week_of_year'].apply(lambda x: np.sin( x * ( 2 * np.pi / 52 ) ) )
        df6['week_of_year_cos'] = df6['week_of_year'].apply(lambda x: np.cos( x * ( 2 * np.pi / 52 ) ) )
        
        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year', 'promo2', 'promo2_since_week',
        'promo2_since_year', 'competition_time_month', 'promo_time_week', 're_competition_distance', 're_competition_time_month', 're_promo_time_week', 'day_of_week_sin', 'day_of_week_cos',
        'day_sin', 'day_cos', 'month_sin', 'month_cos', 'week_of_year_sin', 'week_of_year_cos']
        
        return df6[cols_selected]
    
    def prediction( self, model, original_data, test_data ):
        # prediction
        pred = model.predict(test_data)
        
        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient='records', date_format='iso')
