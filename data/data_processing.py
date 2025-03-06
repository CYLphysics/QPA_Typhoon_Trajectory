import pandas as pd
import os
import math
import numpy as np
from geopy.distance import great_circle

# part 1
# 3D_Typhoon_features_constructor

paths = ['ERA_Interim/z']
pre = 1 # Extract the previous 6, 12, 18, and 24 hours.
new_path = None

for pre in [1, 2, 3, 4]:
    for path in paths:    
        files = os.listdir(path)
        files.sort()
        for file in files:
            analysis = pd.read_csv(path+'/'+file, header=None)
            for tid in analysis[4].unique():
                indexes = analysis[analysis[4]==tid].index
                for i in indexes:
                    if i + pre <= indexes[-1]:
                        analysis.at[i, 0] = analysis.at[i+pre, 0]
                        analysis.at[i, 1] = analysis.at[i+pre, 1]
                        analysis.at[i, 2] = analysis.at[i+pre, 2]
                        analysis.at[i, 3] = analysis.at[i+pre, 3]
                    else:
                        analysis.drop(index=i, inplace=True)
            # 删除后重置索引
            analysis.reset_index(drop=True, inplace=True)
            analysis[[0, 1]] = analysis[[0, 1]] .astype(str)
            # 保存到新目录
            new_path = path+'_'+str(pre*6)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            analysis.to_csv(new_path+'/'+file, header=False, index=False)
            print(file+' is done!')
            
            
# part 2
# CMA_preprocessing


# definition
forward_seq = 4
backward_seq = 4
valid_tropical_seq_num =4
predict_seq_num = 4
trainYear = (2000, 2014)
testYear = (2015, 2018)


"""
typhoonYear
tid Typhoon ID
typhoonRecordNum 
typhoonName 
typhoonRecords 
"""
class TyphoonHeader:

    def __init__(self, typhoonYear, tid):
        self.typhoonYear = typhoonYear
        self.tid = tid
        self.typhoonRecordNum = 0
        self.typhoonRecords = []

    def printer(self):
        print("tid: %d, typhoonYear: %d, typhoonYear: %s, typhoonRecordNum: %d" %
              (self.tid, self.typhoonYear, self.typhoonName, self.typhoonRecordNum))
        for typhoonRecord in self.typhoonRecords:
            typhoonRecord.printer()
            
# Typhoon Record Information Class
"""
typhoonTime Tyhoon Record Time
lat 
long 
wind 
totalNum Typhoon Record ID
"""
class TyphoonRecord:

    def __init__(self, typhoonTime, lat, long, wind, totalNum):
        self.typhoonTime = typhoonTime
        self.lat = lat
        self.long = long
        self.wind = wind
        self.totalNum = totalNum

    def printer(self):
        print("totalNum: %d, typhoonTime: %d, lat: %d, long: %d, wind: %d" %
              (self.totalNum, self.typhoonTime, self.lat, self.long, self.wind))



def get_vector_angle(vector1, vector2):
    a_abs = math.sqrt(vector1[0]*vector1[0] + vector1[1] * vector1[1])
    b_abs = math.sqrt(vector2[0]*vector2[0] + vector2[1] * vector2[1])
    a_b = vector1[0]*vector2[0] + vector1[1]*vector2[1]
    cos_angle = a_b/(a_abs*b_abs)
    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1
    angle = np.arccos(cos_angle) * 360 / 2 / np.pi
    return angle


def read_cma_dfs(cma_list):
    totalNum = 1
    typhoonHeaderList = []
    for df in cma_list:
        tid = df.loc[0, 'TID']
        typhoonYear = df.loc[0, 'YEAR']
        typhoonHeader = TyphoonHeader(typhoonYear, tid, )
        for i in range(len(df)):
            typhoonTime = int(
                str(df.loc[i, 'YEAR'])+
                str(df.loc[i, 'MONTH']).zfill(2)+
                str(df.loc[i, 'DAY']).zfill(2)+
                str(df.loc[i, 'HOUR']).zfill(2)
            )
            lat = df.loc[i, 'LAT'] * 0.1
            long = df.loc[i, 'LONG'] * 0.1
            wind = df.loc[i, 'WND']
            typhoonRecord = TyphoonRecord(typhoonTime, lat, long, wind, totalNum)
            totalNum += 1
            typhoonHeader.typhoonRecords.append(typhoonRecord)
        typhoonHeader.typhoonRecordNum = len(typhoonHeader.typhoonRecords)
        typhoonHeaderList.append(typhoonHeader)
    return typhoonHeaderList



def buildup_feature(typhoonRecordsList, tid, fileXWriter, pred_num):
    for start in range(forward_seq, len(typhoonRecordsList) - predict_seq_num):
        strXLine = str(typhoonRecordsList[start].totalNum) + \
                   ',' + str(tid) + \
                   ',' + str(typhoonRecordsList[start].typhoonTime) + \
                   ',' + str(typhoonRecordsList[start + pred_num].lat) + \
                   ',' + str(typhoonRecordsList[start + pred_num].long) + \
                   ',' + str(typhoonRecordsList[start].typhoonTime//1000000) + \
                   ',' + str(typhoonRecordsList[start].lat) + \
                   ',' + str(typhoonRecordsList[start].long) + \
                   ',' + str(typhoonRecordsList[start].wind)

        # Latitude of the data from the previous 24 hours.
        strXLine += ',' + str(typhoonRecordsList[start - 1].lat)
        strXLine += ',' + str(typhoonRecordsList[start - 2].lat)
        strXLine += ',' + str(typhoonRecordsList[start - 3].lat)
        strXLine += ',' + str(typhoonRecordsList[start - 4].lat)
        
        # Longitude of the data from the previous 24 hours.
        strXLine += ',' + str(typhoonRecordsList[start - 1].long)
        strXLine += ',' + str(typhoonRecordsList[start - 2].long)
        strXLine += ',' + str(typhoonRecordsList[start - 3].long)
        strXLine += ',' + str(typhoonRecordsList[start - 4].long)
        
        # Wind speed of the data from the previous 24 hours
        strXLine += ',' + str(typhoonRecordsList[start - 1].wind)
        strXLine += ',' + str(typhoonRecordsList[start - 2].wind)
        strXLine += ',' + str(typhoonRecordsList[start - 3].wind)
        strXLine += ',' + str(typhoonRecordsList[start - 4].wind)

        # Add month
        strXLine += ',' + str(typhoonRecordsList[start].typhoonTime//10000 % 100)
        
        # Wind speed difference
        strXLine += ',' + str(typhoonRecordsList[start].wind - typhoonRecordsList[start - 1].wind)
        strXLine += ',' + str(typhoonRecordsList[start - 1].wind - typhoonRecordsList[start - 2].wind)
        strXLine += ',' + str(typhoonRecordsList[start - 2].wind - typhoonRecordsList[start - 3].wind)
        strXLine += ',' + str(typhoonRecordsList[start - 3].wind - typhoonRecordsList[start - 4].wind)

        # Latitude difference 0-6, 0-12,0-18,0-24
        latdiff = []
        latdiff.append((typhoonRecordsList[start].lat - typhoonRecordsList[start - 1].lat))
        strXLine += ',' + str(latdiff[-1])
        latdiff.append((typhoonRecordsList[start - 1].lat - typhoonRecordsList[start - 2].lat))
        strXLine += ',' + str(latdiff[-1])
        latdiff.append((typhoonRecordsList[start - 2].lat - typhoonRecordsList[start - 3].lat))
        strXLine += ',' + str(latdiff[-1])
        latdiff.append((typhoonRecordsList[start - 3].lat - typhoonRecordsList[start - 4].lat))
        strXLine += ',' + str(latdiff[-1])

        # Longitude difference
        longdiff = []
        longdiff.append((typhoonRecordsList[start].long - typhoonRecordsList[start - 1].long))
        strXLine += ',' + str(longdiff[-1])
        longdiff.append((typhoonRecordsList[start - 1].long - typhoonRecordsList[start - 2].long))
        strXLine += ',' + str(longdiff[-1])
        longdiff.append((typhoonRecordsList[start - 2].long - typhoonRecordsList[start - 3].long))
        strXLine += ',' + str(longdiff[-1])
        longdiff.append((typhoonRecordsList[start - 3].long - typhoonRecordsList[start - 4].long))
        strXLine += ',' + str(longdiff[-1])

        # Square root of the sum of squared latitude differences
        sum = 0
        for i in range(len(latdiff)):
            sum += latdiff[i]**2
        strXLine += ',' + str(sum)
        strXLine += ',' + str("{:.4f}".format(math.sqrt(sum)))

        # Square root of the sum of squared longitude differences
        sum = 0
        for i in range(len(latdiff)):
            sum += longdiff[i] ** 2
        strXLine += ',' + str(sum)
        strXLine += ',' + str("{:.4f}".format(math.sqrt(sum)))

        # Physical acceleration
        strXLine += ',' + str("{:.4f}".format(
            2 * great_circle(
                (typhoonRecordsList[start].lat, typhoonRecordsList[start].long),
                (typhoonRecordsList[start - 1].lat,typhoonRecordsList[start - 1].long)
            ).kilometers / (6 ** 2)
        ))
        strXLine += ',' + str("{:.4f}".format(
            2 * great_circle(
                (typhoonRecordsList[start - 1].lat,typhoonRecordsList[start - 1].long),
                (typhoonRecordsList[start - 2].lat,typhoonRecordsList[start - 2].long)
            ).kilometers / (6 ** 2)
        ))
        strXLine += ',' + str("{:.4f}".format(
            2 * great_circle(
                (typhoonRecordsList[start].lat,typhoonRecordsList[start].long),
                (typhoonRecordsList[start - 2].lat,typhoonRecordsList[start - 2].long)
            ).kilometers / (12 ** 2)
                                            ))
        strXLine += ',' + str("{:.4f}".format(
            2 * great_circle(
                (typhoonRecordsList[start -2].lat,typhoonRecordsList[start -2].long),
                (typhoonRecordsList[start - 4].lat,typhoonRecordsList[start - 4].long)
            ).kilometers / (12 ** 2)
        ))

        # Square root of the current latitude
        strXLine += ',' + str("{:.4f}".format(math.sqrt(typhoonRecordsList[start].lat)))

        # Square root of the current longitude
        strXLine += ',' + str("{:.4f}".format(math.sqrt(typhoonRecordsList[start].long)))

        ########################################################################################################

        # latitude direction acceleration
        strXLine += ',' + str(
            (typhoonRecordsList[start].lat - typhoonRecordsList[start - 1].lat) -
            (typhoonRecordsList[start - 1].lat - typhoonRecordsList[start - 2].lat)
        )

        strXLine += ',' + str(
            (typhoonRecordsList[start - 2].lat - typhoonRecordsList[start - 3].lat) -
            (typhoonRecordsList[start - 3].lat - typhoonRecordsList[start - 4].lat)
        )
        
        # longitude direction acceleration
        strXLine += ',' + str(
            (typhoonRecordsList[start].long - typhoonRecordsList[start - 1].long) -
            (typhoonRecordsList[start - 1].long - typhoonRecordsList[start - 2].long)
        )

        strXLine += ',' + str(
            (typhoonRecordsList[start - 2].long - typhoonRecordsList[start - 3].long) -
            (typhoonRecordsList[start - 3].long - typhoonRecordsList[start - 4].long)
        )
        
        # Calculate the latitude angle
        for i in range(1, forward_seq + 1):
            diff_lat = typhoonRecordsList[start - i + 1].lat - typhoonRecordsList[start - i].lat
            diff_long = typhoonRecordsList[start - i + 1].long - typhoonRecordsList[start - i].long
            vector1 = [diff_lat, diff_long]
            vector2 = [1, 0]
            if diff_lat < 0 and diff_long < 0:
                strXLine += ',' + str(90 + get_vector_angle(vector1, vector2))
            elif diff_lat > 0 and diff_long < 0:
                strXLine += ',' + str(270 + get_vector_angle(vector1, vector2))
            elif diff_lat == 0 and diff_long == 0:
                strXLine += ',' + str(0)
            else:
                strXLine += ',' + str(get_vector_angle(vector1, vector2))

        # Calculate the longitude angle
        for i in range(1, forward_seq + 1):
            diff_lat = typhoonRecordsList[start - i + 1].lat - typhoonRecordsList[start - i].lat
            diff_long = typhoonRecordsList[start - i + 1].long - typhoonRecordsList[start - i].long
            vector1 = [diff_lat, diff_long]
            vector2 = [0, 1]
            if diff_lat > 0 and diff_long < 0:
                strXLine += ',' + str(90 + get_vector_angle(vector1, vector2))
            elif diff_lat > 0 and diff_long > 0:
                strXLine += ',' + str(270 + get_vector_angle(vector1, vector2))
            elif diff_lat == 0 and diff_long == 0:
                strXLine += ',' + str(0)
            else:
                strXLine += ',' + str(get_vector_angle(vector1, vector2))
        
        # Calculate the angle between two tropical cyclone paths
        for i in range(1, forward_seq):
            diff_lat1 = typhoonRecordsList[start - i + 1].lat - typhoonRecordsList[start - i].lat
            diff_long1 = typhoonRecordsList[start - i + 1].long - typhoonRecordsList[start - i].long
            vector1 = [diff_lat1, diff_long1]
            diff_lat2 = typhoonRecordsList[start - i - 1].lat - typhoonRecordsList[start - i].lat
            diff_long2 = typhoonRecordsList[start - i - 1].long - typhoonRecordsList[start - i].long
            vector2 = [diff_lat2, diff_long2]
            if diff_lat1 == 0 and diff_long1 == 0 or diff_lat2 == 0 and diff_long2 == 0:
                strXLine += ',' + str(0)
            else:
                strXLine += ',' + str(get_vector_angle(vector1, vector2))

        fileXWriter.write(strXLine + '\n')
        
        

# Read CMA 

path = "./CMA"
files = os.listdir(path)
files.sort()

pd_list = []
for file in files:
    cma_pd = pd.read_csv(path+'//'+file, delim_whitespace=True, 
                         names=['TROPICALTIME', 'I', 'LAT', 'LONG', 'PRES', 'WND' , 'OWD', 'NAME', 'RECORDTIME'])
    pd_list.append(cma_pd)

df = pd.concat(pd_list, axis=0)
df = df.reset_index(drop=True)

# CMA data 

df = df.drop(columns=['OWD','RECORDTIME'])
df = pd.concat([df, pd.DataFrame(columns=['TID','YEAR','MONTH','DAY','HOUR'])], axis=1)
df = df[['TID','YEAR','MONTH','DAY','HOUR','TROPICALTIME', 'I', 'LAT', 'LONG', 'WND', 'PRES', 'NAME']]
tid = 0
name = None
for i in range(0, len(df)):
    if df.at[i, 'TROPICALTIME'] == 66666:
        tid += 1
        name = df.loc[i, 'NAME']
    else:
        df.at[i, 'TID'] = tid
        df.at[i, 'NAME'] = name
        df.at[i, 'YEAR'] = df.loc[i, 'TROPICALTIME'] // 1000000
        df.at[i, 'MONTH'] = df.loc[i, 'TROPICALTIME'] // 10000 % 100
        df.at[i, 'DAY'] = df.loc[i, 'TROPICALTIME'] // 100 % 100
        df.at[i, 'HOUR'] = df.loc[i, 'TROPICALTIME'] % 100

df = df.drop(df[df['TROPICALTIME']==66666].index, axis=0)
df = df.drop(columns=['TROPICALTIME'])
df = df.reset_index(drop=True)

df.loc[df['NAME']=='In-fa', 'NAME'] = 'Infa'
df[df['NAME']=='Infa'] 


# Add a "KEY" column
df['KEY'] = None

# Create a dictionary to track records by year
years = df['YEAR'].unique()
years_dict = dict(zip(years, np.ones(years.shape)))  # Initialize each year with 1

# Store the computed results
result_list = []

# Calculate the typhoon sequence number for each year
for tid in df['TID'].unique():
    temp_df = df[df['TID'] == tid].copy()
    
    # Assign the typhoon to the previous year if it spans multiple years
    tid_year = temp_df['YEAR'].unique()[0]
    cy = int(years_dict[tid_year])  # Get the current typhoon count for that year
    years_dict[tid_year] += 1  # Increment the count
    
    # Generate a unique KEY in the format "YEAR-XX" (e.g., "2023-01")
    temp_df['KEY'] = str(tid_year) + '-' + str(cy).zfill(2)
    
    result_list.append(temp_df)

# Concatenate the results into a single DataFrame
df = pd.concat(result_list, axis=0)

# Reset index after merging
df = df.reset_index(drop=True)



# Unique 

df = df.drop(df[~df['HOUR'].isin([0,6,12,18])].index, axis=0)
df = df.reset_index(drop=True)
df = df.drop_duplicates()
df = df.reset_index(drop=True)

df.to_csv('./raw.csv', index=False)

print('raw.csv sucessfully saved.')

# Data preprocessing 

# Split each typhoon into a separate DataFrame

df = pd.read_csv('./raw.csv')
tids = df['TID'].unique()
cma_list = []
for tid in tids:
    temp_df = df[df['TID']==tid]
    temp_df = temp_df.reset_index(drop=True)
    cma_list.append(temp_df)
    
valid_tropical_len = forward_seq + valid_tropical_seq_num + backward_seq

temp = []
for df in cma_list:
    if df.shape[0] >= valid_tropical_len:
        temp.append(df)
        
cma_list = temp

df = pd.concat(cma_list, axis=0)
df=df.reset_index(drop=True)
train_range = [ str(x) for x in range(trainYear[0], trainYear[1]+1) ]
train_keys =  [v for i, v in enumerate(df['KEY'].unique()) if any(s in v for s in train_range)]

test_range = [ str(x) for x in range(testYear[0], testYear[1]+1) ]
test_keys =  [v for i, v in enumerate(df['KEY'].unique()) if any(s in v for s in test_range)]

df = df[(df['KEY'].isin(train_keys)) | (df['KEY'].isin(test_keys))]
df=df.reset_index(drop=True)

tname = pd.read_csv('./typhoon_name.csv')

dict_name = {}
for i in range(len(tname)):
    dict_name[tname.at[i, 'en'].lower()] = tname.at[i, 'cn']
dict_name['(nameless)']='nameless'

df['CN_NAME'] = None
for i in range(len(df)):
    try:
        df.at[i, 'CN_NAME'] = dict_name[df.at[i, 'NAME'].lower()]
    except KeyError:
        print(df.at[i, 'NAME'].lower())

df.to_csv('./pre_processing.csv', index=False)

typhoonHeaderList = read_cma_dfs(cma_list)

for i in range(1, predict_seq_num+1):
    trainXFile = open('./CMA_train_'+str(i*6)+'h.csv', 'w')
    testXFile = open('./CMA_test_'+str(i*6)+'h.csv', 'w')
    for typhoonHeader in typhoonHeaderList:
        typhoonRecordsList = typhoonHeader.typhoonRecords
        if typhoonHeader.typhoonYear in range(trainYear[0], trainYear[1]+1):
            buildup_feature(typhoonRecordsList, typhoonHeader.tid, trainXFile, i)
        elif typhoonHeader.typhoonYear in range(testYear[0], testYear[1]+1):
            buildup_feature(typhoonRecordsList, typhoonHeader.tid, testXFile, i)

    trainXFile.close()
    testXFile.close()
    
cma_ecwmf_train = open('./cma_ecwmf_train.csv', 'w')
cma_ecwmf_test = open('./cma_ecwmf_test.csv', 'w')

for typhoonHeader in typhoonHeaderList:
    typhoonRecordsList = typhoonHeader.typhoonRecords
    for start in range(0, len(typhoonRecordsList) - 4):
        strXLine = str(typhoonRecordsList[start].totalNum) + \
           ',' + str(typhoonRecordsList[start].typhoonTime) + \
           ',' + str(typhoonRecordsList[start].lat) + \
           ',' + str(typhoonRecordsList[start].long) + \
           ',' + str(typhoonHeader.tid)
        if typhoonHeader.typhoonYear in range(trainYear[0], trainYear[1]+1):
            cma_ecwmf_train.write(strXLine + '\n')
        elif typhoonHeader.typhoonYear in range(testYear[0], testYear[1]+1):
            cma_ecwmf_test.write(strXLine + '\n')

cma_ecwmf_train.close()
cma_ecwmf_test.close()