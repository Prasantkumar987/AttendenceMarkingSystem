import pandas as pd

df= pd.read_csv('attendence/2022-01-06.csv')
df.loc[df.E_Id== int('00001'), "Attendence_Punching_time"] = "18:48:06"
print(df)