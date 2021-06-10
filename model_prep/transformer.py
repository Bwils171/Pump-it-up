
def model_transformer_train(model_data):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    #Applies transformations from EDA notebook to training and testing sets to ensure same changes are made
    #Correct names in extraction_type
    data = model_data.copy()
    
    #Fill NaN values
    data.fillna(inplace=True, value={'installer':'unknown','permit':False, 'funder':'unknown', 'public_meeting':False, 
                                 'scheme_management':'unknown', 'scheme_name':'unknown', 'subvillage':'unknown'})
    
    #WATER AND EQUIPMENT INFORMATION
    #correct names in source
    data['source'].replace({'shallow well':'shallow_well',
                       'machine dbh':'machine_dbh',
                       'rainwater harvesting':'rainwater_harvesting',
                       'hand dtw':'hand_dtw'}, inplace=True)
    
    #correct extraction category names
    data['extraction_type'].replace({'other - swn 81':'other-handpump',
                                 'other - play pump':'other-handpump', 
                                 'walimi':'other-handpump', 
                                 'other - mkulima/shinyanga':'other-handpump',
                                'swn 80':'swn_80',
                                 'nira/tanira':'nira-tanira',
                                'india mark ii':'india_mark_ii',
                                'india mark iii':'india_mark_iii',
                                'other - rope pump':'other-rope_pump',}, inplace=True)
    
    #Create column for water quality and quantity and combination column of these two ranks
    data['water_quality_rank'] = data['water_quality'].map({'fluoride':7, 'soft':6, 'coloured':5,'milky':4, 
                                                        'salty':3, 'salty abandoned':2, 'fluoride abandoned':1, 
                                                       'unknown':0})
    data['quantity_rank'] = data['quantity'].map({'enough':4, 'seasonal':3, 'insufficient':2,'unknown':1, 'dry':0})
    data['quant_qual_rank']=data['quantity_rank']+data['water_quality_rank']
    
    
    #LOCATION INFORMATION
    #Latitude-Longitiude - Correct near zero values
    data.loc[data['longitude'] == 0, 'longitude'] = np.random.choice(range(31,33))
    data.loc[data['latitude']>-0.01, 'latitude'] = -1*np.random.choice(range(1,2))
    
    #Correct subvillage names with 3 or less characters based on KNeigborsRegressor and latitiude/longitude
    sv_trainX = data.loc[data['subvillage'].str.len()>3, ['latitude', 'longitude']]
    sv_trainy = data.loc[data['subvillage'].str.len()>3, ['subvillage']]
    knn_sv = KNeighborsClassifier(n_neighbors=2, weights='distance', metric='euclidean')
    knn_sv.fit(sv_trainX, sv_trainy)
    data.loc[data['subvillage'].str.len()<=3, ['subvillage']] = knn_sv.predict(data.loc[data['subvillage'].str.len()<=3, 
                                                                           ['latitude', 'longitude']])  
    
    #Create column for size of subvillage based on number of wellpoints
    counts = data['subvillage'].value_counts()
    verybig = counts.loc[counts.between(400,515)].index
    big = counts.loc[counts.between(100,400)].index
    medium = counts.loc[counts.between(50,100)].index
    small = counts.loc[counts.between(25,50)].index
    verysmall = counts.loc[counts.between(15,25)].index
    smaller = counts.loc[counts.between(5,15)].index
    smallest = counts.loc[counts.between(2,5)].index
    single = counts.loc[counts ==1].index
    data.loc[data['subvillage'].isin(verybig), 'subvillage_rank'] = 7
    data.loc[data['subvillage'].isin(big), 'subvillage_rank'] = 6
    data.loc[data['subvillage'].isin(medium), 'subvillage_rank'] = 5
    data.loc[data['subvillage'].isin(small), 'subvillage_rank'] = 4
    data.loc[data['subvillage'].isin(verysmall), 'subvillage_rank'] = 3
    data.loc[data['subvillage'].isin(smaller), 'subvillage_rank'] = 2
    data.loc[data['subvillage'].isin(smallest), 'subvillage_rank'] = 1
    data.loc[data['subvillage'].isin(single), 'subvillage_rank'] = 0
    
    #Group low count subvillages in other
    counts = data['subvillage'].value_counts()
    counts = counts.loc[counts >=90]
    counts = list(counts.index)
    data.loc[~data['subvillage'].isin(counts), 'subvillage'] = 'other'
        
    #create and urban_rural columns based on lga labels, and reduce lga categories
    data['urban_rural'] = 'unknown'
    data.loc[data['lga'].str.contains('Rural|rural'), 'urban_rural'] = 'rural'
    data.loc[data['lga'].str.contains('Urban|urban'), 'urban_rural'] = 'urban'
    counts = data['lga'].value_counts()
    counts = counts.loc[counts >=500]
    counts = list(counts.index)
    data.loc[~data['lga'].isin(counts), 'lga'] = 'other'

    #remove slashes from basin names
    data['basin'].replace({'Ruvuma / Southern Coast':'Ruvuma-Southern_Coast',
                     'Wami / Ruvu':'Wami-Ruvu'}, inplace=True)
    
    #Change ward column to categorical denoting size based on amount of wellpoints
    counts = data['ward'].value_counts()
    verybig = counts.loc[counts.between(200,400)].index
    big = counts.loc[counts.between(100,200)].index
    medium = counts.loc[counts.between(50,100)].index
    small = counts.loc[counts.between(25,50)].index
    verysmall = counts.loc[counts.between(15,25)].index
    smaller = counts.loc[counts.between(5,15)].index
    smallest = counts.loc[counts.between(2,5)].index
    single = counts.loc[counts ==1].index
    data.loc[data['ward'].isin(verybig), 'ward_rank'] = 7
    data.loc[data['ward'].isin(big), 'ward_rank'] = 6
    data.loc[data['ward'].isin(medium), 'ward_rank'] = 5
    data.loc[data['ward'].isin(small), 'ward_rank'] = 4
    data.loc[data['ward'].isin(verysmall), 'ward_rank'] = 3
    data.loc[data['ward'].isin(smaller), 'ward_rank'] = 2
    data.loc[data['ward'].isin(smallest), 'ward_rank'] = 1
    data.loc[data['ward'].isin(single), 'ward_rank'] = 0
    
    #Reduce number of ward categories
    counts = data['ward'].value_counts()
    counts = counts.loc[counts >=100]
    counts = list(counts.index)
    data.loc[~data['ward'].isin(counts), 'ward'] = 'other'

        
    #DATE/TIME INFORMATION
    #convert date_recorded column to datetime object and edxtract month and year
    data['date_recorded']= pd.to_datetime(data['date_recorded'])
    data['date_recorded'].describe(datetime_is_numeric=True)
    data['year']=data['date_recorded'].dt.year
    data['month']=data['date_recorded'].dt.month
    
    #inpute construction_year for 0 values based on KNeighborRegressor and some columns
    used = ['latitude', 'longitude', 'extraction_type', 'source', 'waterpoint_type']
    data.loc[data['construction_year']==0, 'construction_year'] = 1950
    data['construction_year'] = pd.to_datetime(data['construction_year'], format='%Y')
    data['construction_year'] = data['construction_year'].dt.year
    old_X = data.loc[data['construction_year']!=1950, used]
    old_X_dum = pd.get_dummies(old_X)
    old_y = data.loc[data['construction_year']!=1950, ['construction_year']]
    knn_old = KNeighborsRegressor(n_neighbors=1)
    knn_old.fit(old_X_dum, old_y)
    predictor = pd.get_dummies(data.loc[data['construction_year']==1950, used])
    data.loc[data['construction_year']==1950, ['construction_year']] = knn_old.predict(predictor)
    data['construction_year'] = data['construction_year'].astype('int')
    data['years_old'] = data['date_recorded'].dt.year - data['construction_year']
        
    
    #ADMINISTRATIVE INFORMATION
    ##Convert public_meeting column to 1 or 0
    data['public_meeting'] = data['public_meeting'].map({True:1, False:0})

    #Convert permit column to 1 or 0
    data['permit'] = data['permit'].map({True:1, False:0})

    #Group low count scheme_names under other to reduce categories
    counts2 = data['scheme_name'].value_counts()
    counts2 = counts2.loc[counts2 >=200]
    counts2 = list(counts2.index)
    data.loc[~data['scheme_name'].isin(counts2), 'scheme_name'] = 'other'
    
    #group low count funders under other to reduce categories
    counts3 = data['funder'].value_counts()
    counts3 = counts3.loc[counts3 >=500]
    counts3 = list(counts3.index)
    data.loc[~data['funder'].isin(counts3), 'funder'] = 'other'
    data.loc[data['funder']=='Government Of Tanzania', 'funder'] = 'gov_tanz'

    #Group low count installers under other to reduce categories
    counts4 = data['installer'].value_counts()
    counts4 = counts4.loc[counts4 >=500]
    counts4 = list(counts4.index)
    data.loc[~data['installer'].isin(counts4), 'installer'] = 'other'
    
    #Create boolean column of whether funder also built the wellpoint
    data['fund_inst_same'] = 0
    data.loc[data['funder']==data['installer'],['fund_inst_same']] = 1
    
    #Create ranked column for payment type
    data['pay_rank'] = data['payment_type'].map({'annually':6, 'per bucket':5, 'monthly':4,'on failure':3, 'other':2, 'never pay':1, 'unknown':0})
    
    #NUMERIC INFORMATION
    #Create column for population bins
    data['popbins'] = pd.cut(data['population'], [-1,2,250,500,1000,2500,10000,400000], labels=list(range(1,8)))
    
    #GPS height inputation based on KNeighborsRegressor with latitude and longitude
    gps_trainX = data.loc[data['gps_height']>0, ['latitude', 'longitude']]
    gps_trainy = data.loc[data['gps_height']>0, ['gps_height']]
    knn_gps = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='euclidean')
    knn_gps.fit(gps_trainX, np.ravel(gps_trainy))
    data.loc[data['gps_height']<=0, ['gps_height']] = knn_gps.predict(data.loc[data['gps_height']<=0, 
                                                                           ['latitude', 'longitude']])
    
    #Correct any amount_tsh values over 500 to a max of 500 and create categorical column
    col = ['gps_height', 'latitude', 'longitude', 'quant_qual_rank']
    data['tsh_rank'] = pd.cut(data['amount_tsh'], bins=[-1,0,5,10,25,50,100,250,500,400000], labels=['error_low', 'limited', 
                                             'small', 'ok', 'medium', 'good', 'great', 'grand', 'error_high'])
    tsh_X = data.loc[~data['tsh_rank'].isin(['error_low', 'error_high']), col]
    tsh_y = data.loc[~data['tsh_rank'].isin(['error_low', 'error_high']), ['tsh_rank']]

    rf_tsh = RandomForestClassifier(n_estimators=250)
    rf_tsh.fit(tsh_X, np.ravel(tsh_y))
    data.loc[data['tsh_rank'].isin(['error_low', 'error_high']), ['tsh_rank']] = rf_tsh.predict(data.loc[data['tsh_rank'].isin(['error_low', 'error_high']), col])
    data.loc[data['amount_tsh']>500, 'amount_tsh'] = 500
    data['amount_tsh']=data['amount_tsh'].astype('int')
    
    #drop any columns that are deemed duplicates or not useful based on notebook EDA   
    data.drop(columns=['source_type', 'source_class', 'extraction_type_group', 'extraction_type_class', 
                       'region', 'wpt_name', 'num_private', 'recorded_by', 'quality_group', 'quantity_group',
                       'waterpoint_type_group', 'payment', 'construction_year', 'date_recorded'], inplace=True)
    return data
