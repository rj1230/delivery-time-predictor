#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("Food delivery.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


type(df["Delivery_person_Age"][0])


# In[7]:


df["Weatherconditions"].value_counts()


# In[8]:


df["Weatherconditions"] = df["Weatherconditions"].apply(
    lambda temp: temp.split(" ")[1] if pd.notnull(temp) else temp
)


# In[9]:


df["Weatherconditions"]


# In[10]:


df.columns


# In[11]:


df.select_dtypes(include="object").columns


# In[12]:


for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Standardize common textual missing-value markers
missing_values = ["NaN", "nan", "NAN", "NA", "N/A", "null", "NULL", "None", ""]

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].replace(missing_values, np.nan)


# In[13]:


df["Delivery_person_Age"].value_counts()


# In[14]:


df[df["Delivery_person_Age"]=='NaN']


# In[15]:


df["Time_taken(min)"]


# In[16]:


df["Time_taken(min)"] = df["Time_taken(min)"].apply(lambda temp : int(temp.split(" ")[1]) if pd.notnull(temp) else temp)


# In[17]:


df["Time_taken(min)"]


# In[18]:


df.head()


# In[19]:


for col in ['Delivery_person_Age','Delivery_person_Ratings','multiple_deliveries'] :
  df[col]=pd.to_numeric(df[col],errors='coerce')


# In[20]:


df.info()


# In[21]:


df[df["Delivery_person_Ratings"]>5]["Delivery_person_Ratings"].shape


# In[22]:


df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].clip(upper=5)


# In[23]:


df.head()


# In[24]:


df.describe()


# In[25]:


# Numeric columns: fill missing values with the median (robust to outliers/skew)
for col in ['Delivery_person_Age','Delivery_person_Ratings','multiple_deliveries'] :
  df[col]=df[col].fillna(df[col].median())


# In[26]:


# Categorical columns: fill missing values with the mode (most frequent category)
for col in ['Weatherconditions','Road_traffic_density','Festival','City']:
  df[col].fillna(df[col].mode()[0],inplace=True)


# In[27]:


df.info()


# In[28]:


df["Road_traffic_density"].value_counts()


# In[29]:


sns.histplot(df['Time_taken(min)'],bins=30,kde =True)
plt.title("distribution of delivery time")
plt.show()


# In[30]:


df.info()


# In[31]:


df.groupby('Road_traffic_density')['Time_taken(min)'].agg(['count','mean','median','std'])


# In[32]:


df.groupby('Weatherconditions')['Time_taken(min)'].agg(['count','mean','median','std'])


# In[33]:


df.groupby('Festival')['Time_taken(min)'].agg(['count','mean','median','std'])


# In[34]:


df.groupby('City')['Time_taken(min)'].agg(['count','mean'])


# In[35]:


df.groupby('Type_of_vehicle')['Time_taken(min)'].agg(['count','mean'])


# In[36]:


df.groupby('Type_of_order')['Time_taken(min)'].agg(['count','mean'])


# In[37]:


df['Delivery_person_Age'].describe()


# In[38]:


sns.histplot(df['Delivery_person_Age'],bins=10)


# In[39]:


df[df['Delivery_person_Age']>40]['Delivery_person_Age'].shape


# In[40]:


plt.figure(figsize=(10,5))

sns.lineplot(
    data=df,
    x="Delivery_person_Age",
    y="Time_taken(min)",
    estimator="mean",
    errorbar=None,
    marker="o"
)

plt.title("Average Delivery Time vs Delivery Person Age")
plt.xlabel("Delivery Person Age")
plt.ylabel("Average Delivery Time (minutes)")

plt.show()


# In[41]:


df['Delivery_person_Ratings'].describe()


# In[42]:


plt.figure(figsize=(10,5))

sns.lineplot(
    data=df,
    x='Delivery_person_Ratings',
    y="Time_taken(min)",
    estimator="mean",
    errorbar=None,
    marker="o"
)

plt.title("Average Delivery Time vs Delivery Person Age")
plt.xlabel("Delivery Person Rating")
plt.ylabel("Average Delivery Time (minutes)")

plt.show()


# In[43]:


df[df['Delivery_person_Ratings']<4].shape[0]/len(df)


# In[44]:


numeric_cols=['Delivery_person_Age','Delivery_person_Ratings','multiple_deliveries','Vehicle_condition','Time_taken(min)']
sns.heatmap(df[numeric_cols].corr(),annot=True,cmap='coolwarm')
plt.title('Correlation — numeric columns before feature engineering')
plt.show()


# In[45]:


df.info()


# In[46]:


df['Order_Date']


# In[47]:


df['Time_Orderd']


# In[48]:


hour=pd.to_datetime(df['Time_Orderd'],format='%H:%M:%S', errors='coerce').dt.hour


# In[49]:


hour.unique()


# In[50]:


df.groupby(hour)['Time_taken(min)'].agg(['mean','median'])


# In[51]:


df.groupby(hour)['Time_taken(min)'].mean().plot(kind='bar',figsize=(11,4))
plt.title('Average Delivery Time by Order Hour')
plt.ylabel('Avg minutes')
plt.show()


# In[52]:


df.info()


# In[53]:


#Distance from coordinations


# In[54]:


def calculate_series_distance(
    lat1_series, lon1_series, lat2_series, lon2_series, miles=False
):
    """Calculates the great-circle (straight-line) distance between two series of

    lat/lon points using the Haversine formula. Returns distance in km (or
    miles). Turns four raw coordinate columns into one meaningful 'distance'
    feature.
    """
    r = 3956 if miles else 6371
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1_series, lon1_series, lat2_series, lon2_series]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * r


# In[55]:


df['distance_km']=calculate_series_distance(df['Restaurant_latitude'],df['Restaurant_longitude'],df['Delivery_location_latitude'],df['Delivery_location_longitude'])


# In[56]:


df.info()


# In[57]:


df['distance_km'].describe()


# In[58]:


df[df['distance_km']>100]['distance_km'].describe()


# In[59]:


df[df['distance_km']<100]['distance_km'].describe()


# In[60]:


df[df['distance_km']>100]


# In[61]:


df=df[df['distance_km']<100].reset_index(drop=True)


# In[62]:


df.info()


# In[63]:


df['order_hour']=pd.to_datetime(df['Time_Orderd'],format='%H:%M:%S', errors='coerce').dt.hour


# In[64]:


df['order_hour'].fillna(df['order_hour'].median(),inplace=True)


# In[65]:


df.info()


# In[66]:


len(df['Delivery_person_ID'].unique())


# In[67]:


features=['Delivery_person_Age','Delivery_person_Ratings','distance_km','order_hour','Weatherconditions','Road_traffic_density',
          'Vehicle_condition','Type_of_order','Type_of_vehicle','multiple_deliveries','Festival','City']



# In[68]:


X=df[features]


# In[69]:


y=df['Time_taken(min)']


# In[70]:


#Encoding


# In[71]:


traffic_mapping={
    'Low':0,
    'Medium': 1,
    'High' : 2,
    'Jam' : 3
}

X['Road_traffic_density']=X['Road_traffic_density'].map(traffic_mapping)


# In[72]:


X['Road_traffic_density']


# In[73]:


X['Weatherconditions'].value_counts()


# In[74]:


X=pd.get_dummies(X,columns=['Type_of_order','Type_of_vehicle','Festival','City','Weatherconditions'],drop_first=True,dtype=int)


# In[75]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape


# FEATURE SCALING

# In[76]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()


# In[77]:


# Fit the scaler on TRAIN ONLY, then apply it to both — never fit on test data
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


# In[78]:


X_train_scaled


# In[79]:


X_train_scaled=pd.DataFrame(X_train_scaled,columns=X_train.columns,index=X_train.index)
X_test_scaled=pd.DataFrame(X_test_scaled,columns=X_test.columns,index=X_test.index)


# In[80]:


import joblib
# Save the fitted scaler
joblib.dump(scaler, "standard_scaler.pkl")


# In[81]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

lr.fit(X_train, y_train)


# In[82]:


y_pred = lr.predict(X_test)


# In[83]:


import pandas as pd

coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": lr.named_steps["regressor"].coef_
}).sort_values("coefficient", key=abs, ascending=False)

coef_df


# In[84]:


X_train.columns


# In[85]:


from sklearn.metrics import r2_score,mean_squared_error
y_pred_test=lr.predict(X_test_scaled)


# In[86]:


rmse_test=np.sqrt(mean_squared_error(y_test,y_pred_test))


# In[87]:


rmse_train=np.sqrt(mean_squared_error(y_train,lr.predict(X_train_scaled)))


# In[88]:


r2_test=r2_score(y_test,y_pred_test)


# In[89]:


r2_train=r2_score(y_train,lr.predict(X_train_scaled))


# In[90]:


print(rmse_train,r2_train)


# In[91]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=42)


# In[92]:


dt.fit(X_train_scaled,y_train)


# In[93]:


y_pred_dt=dt.predict(X_test_scaled)


# In[94]:


# Test performance
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
mse=mean_squared_error(y_test,dt.predict(X_test_scaled))
rmse=np.sqrt(mse)
r2=r2_score(y_test,dt.predict(X_test_scaled))
print("dt rmse",rmse)
print("dt r2",r2)


# In[95]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
mse=mean_squared_error(y_train,dt.predict(X_train_scaled))
rmse=np.sqrt(mse)
r2=r2_score(y_train,dt.predict(X_train_scaled))
print("dt train rmse",rmse)
print("dt train r2",r2)


# In[96]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,random_state=42)


# In[97]:


rf.fit(X_train_scaled,y_train)


# In[98]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
mse=mean_squared_error(y_test,rf.predict(X_test_scaled))
rmse=np.sqrt(mse)
r2=r2_score(y_test,rf.predict(X_test_scaled))
print("rf rmse",rmse)
print("rf r2",r2)


# In[99]:


mse=mean_squared_error(y_train,rf.predict(X_train_scaled))
rmse=np.sqrt(mse)
r2=r2_score(y_train,rf.predict(X_train_scaled))
print("rf rmse",rmse)
print("rf r2",r2)


# Hyperparameter tunning

# In[100]:


from sklearn.model_selection import RandomizedSearchCV

rf_param_dist={
    'n_estimators':[200,300,400,500],
    'max_depth':[10,15,20,25,None],
     'min_samples_leaf':[1,2,4],
    'min_samples_split':[2,5,10],
    'max_features':['sqrt','log2',0.5]
}
# # Sample 25 random combinations instead of testing every combination exhaustively
rf_search=RandomizedSearchCV(RandomForestRegressor(random_state=42),rf_param_dist,n_iter=25,cv=3,scoring='r2',random_state=42,n_jobs=-1)
rf_search.fit(X_train_scaled,y_train)
print("best params",rf_search.best_params_)
rf_best=rf_search.best_estimator_


# In[101]:


r2_rf_tuned=r2_score(y_test,rf_best.predict(X_test_scaled))
rmse_rf_tuned=np.sqrt(mean_squared_error(y_test,rf_best.predict(X_test_scaled)))
print("tuned rf rmse",rmse_rf_tuned)
print("tuned rf r2",r2_rf_tuned)


# In[102]:


rf.feature_importances_


# In[103]:


X_train.columns


# XGBoost

# In[104]:


from xgboost import XGBRegressor
xgb=XGBRegressor(n_estimators=100,random_state=42)


# In[105]:


xgb.fit(X_train_scaled,y_train)


# In[106]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
mse=mean_squared_error(y_test,xgb.predict(X_test_scaled))
rmse=np.sqrt(mse)
r2=r2_score(y_test,xgb.predict(X_test_scaled))
print("xgb rmse",rmse)
print("xgb r2",r2)


# In[ ]:


xgb_param_dist={
    'n_estimators':[300,500,800],
    'max_depth':[3,4,5,6,8],
    'learning_rate':[0.01,0.03,0.05,0.1],
    'subsample':[0.7,0.8,0.9,1.0],
    'colsample_bytree':[0.7,0.8,0.9,1.0],
    'reg_alpha':[0,0.1,0.5,1],
    'reg_lambda':[1,1.5,2]
}
xgb_search=RandomizedSearchCV(XGBRegressor(random_state=42),xgb_param_dist,n_iter=40,cv=3,scoring='r2',random_state=42,n_jobs=-1)
xgb_search.fit(X_train_scaled,y_train)
print("best params",xgb_search.best_params_)
xgb_best=xgb_search.best_estimator_


# In[ ]:


r2_xgb_train=r2_score(y_train,xgb_best.predict(X_train_scaled))
rmse_xgb_train=np.sqrt(mean_squared_error(y_train,xgb_best.predict(X_train_scaled)))
print("xgb train rmse",rmse_xgb_train)
print("xgb train r2",r2_xgb_train)


# In[ ]:


r2_xgb_test=r2_score(y_test,xgb_best.predict(X_test_scaled))
rmse_xgb_test=np.sqrt(mean_squared_error(y_test,xgb_best.predict(X_test_scaled)))
print("xgb rmse",rmse_xgb_test)
print("xgb r2",r2_xgb_test)


# In[ ]:


X_test_scaled.shape[1]


# In[ ]:


# Test Adjusted R²
y_test_pred = xgb_best.predict(X_test_scaled)

n_test = len(y_test)
p_test = X_test_scaled.shape[1]

adj_r2_xgb_test = 1 - (1 - r2_xgb_test) * (n_test - 1) / (n_test - p_test - 1)

print("xgb test adjusted r2", adj_r2_xgb_test)


# In[ ]:


# Recall: X_train is the UNSCALED training features — we build the new feature here first
X_train


# In[ ]:


def order_period(hour):
  if 12<=hour<=14 :
    return "Lunch"
  elif 19<=hour<=21 :
    return "Dinner"
  else :
    return "Normal"


# In[ ]:


# 1. Copy unscaled data
X_train_exp = X_train.copy()
X_test_exp = X_test.copy()


# In[ ]:


# 2. Create the new feature BEFORE scaling
X_train_exp['order_period'] = X_train_exp['order_hour'].apply(order_period)
X_test_exp['order_period'] = X_test_exp['order_hour'].apply(order_period)


# In[ ]:


# Confirm the new order_period column landed correctly
X_train_exp


# In[ ]:


# 3. One-hot encode
X_train_exp = pd.get_dummies(
    X_train_exp, columns=['order_period'], drop_first=True, dtype=int
)

X_test_exp = pd.get_dummies(
    X_test_exp, columns=['order_period'], drop_first=True, dtype=int
)


# In[ ]:


# 4. Create a NEW scaler and fit ONLY on expanded training data
scaler_exp = StandardScaler()

X_train_exp_scaled = scaler_exp.fit_transform(X_train_exp)
X_test_exp_scaled = scaler_exp.transform(X_test_exp)


# In[ ]:


# Retrain our strongest model (same tuned hyperparameters) WITH the new feature, for a fair comparison
xgb_with_period = XGBRegressor(**xgb_search.best_params_, random_state=42)
xgb_with_period.fit(X_train_exp_scaled, y_train)


# In[ ]:


n_test = X_test_exp.shape[0]
p_test = X_test_exp.shape[1]
r2_with_period_test = r2_score(y_test, xgb_with_period.predict(X_test_exp_scaled))
adj_r2_test = 1 - (1 - r2_with_period_test) * (n_test - 1) / (n_test - p_test - 1)
print("Test Adjusted R²:", adj_r2_test)


# RESULT & MODEL PERSISTENCE

# In[ ]:


joblib.dump(xgb_best, "xgb_best.pkl")


# In[ ]:


model = joblib.load("./xgb_best.pkl")
scaler = joblib.load("./standard_scaler.pkl")


# In[ ]:


sample=X_test[0:3]


# In[ ]:


sample_scaled=scaler.transform(sample)


# In[ ]:


predictions=model.predict(sample_scaled)


# In[ ]:


predictions


# In[ ]:


y_test[0:3].values


# In[ ]:


get_ipython().system('jupyter nbconvert --to script main.ipynb')

