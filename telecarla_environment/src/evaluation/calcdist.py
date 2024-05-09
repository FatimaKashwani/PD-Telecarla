import os
import pandas as pd
import numpy as np


def calc_dist(clientcsv, servercsv):
    # Calculate error between predicted distance and actual distance
    client_df = pd.read_csv(clientcsv)
    server_df = pd.read_csv(servercsv)

    column_names = ['t1_srvr', 't2_srvr', 't1_clnt', 't2_clnt','dist_pred', 'dist_act']
    comb_df = pd.DataFrame(columns=column_names)

    def add_to_df(index_c, index_s1, index_s2):
        nonlocal comb_df 
        print("first index: ", index_s1)
        print("second index: ", index_s2)

        # use averaged speed to calculate how far the vehicle has moved
        sum = 0
        count = 0
        for i in range(index_s2 - index_s1):
            value = server_df['speed'][index_s1 + i]
            if pd.notnull(value):  
                numeric_value = pd.to_numeric(value, errors='coerce')
                if not np.isnan(numeric_value):
                    sum += numeric_value
                    count += 1

        if count > 0:
            sum = sum / count
        else:
            sum = None

        # Get actual distance = avg speed * time elapsed
        dist_act = abs(sum*(abs(server_df['time'][index_s1] - server_df['time'][index_s2])))
        
        
        print("distance: ", dist_act)
        if dist_act == 0: return
        

        # Use this if statement to only consider CONSTANT SPEED samples
        #if abs(server_df['speed'][index_s1])>=28.5 and abs(server_df['speed'][index_s2])>=28.5:
        new_row = {'t1_srvr': server_df['time'][index_s1],
                   't2_srvr':server_df['time'][index_s2],
                   't1_clnt':client_df['time 1'][index_c],
                   't2_clnt':client_df['time 2'][index_c],
                   'dist_pred':client_df['dist'][index_c],
                   'dist_act':dist_act}
        print(new_row)
        comb_df = pd.concat([comb_df, pd.DataFrame([new_row])], ignore_index=True) 


    pos = 0
    #iterate to get matching timestamps in the combined dataframe
    for index in range (len(client_df)):
        t1 = client_df['time 1'][index]
        print("index "+str(index)+" of "+str(len(client_df)))
        found = False
        time = server_df['time'][index]
        if t1 > time:
            newindex = index
            while t1 > server_df['time'][newindex] and newindex + 1 < len(server_df):
                newindex += 1
                if t1 == server_df['time'][newindex]:
                    found = True
                    pos = newindex
        if t1 == time:
            found = True
            pos = index
        if t1 < time:
            newindex = pos
            while t1 > server_df['time'][newindex]:
                newindex += 1
                if t1 == server_df['time'][newindex]:
                    found = True
                    pos = newindex
        if found:
            if pos + 1 < len(server_df):
                if server_df['time'][pos + 1] == client_df['time 2'][index]:
                    add_to_df(index, pos, pos+1) 
                elif server_df['time'][pos + 1] < client_df['time 2'][index]:
                    add = 1
                    while server_df['time'][pos + add] < client_df['time 2'][index]:
                        add += 1
                        if pos+add >= len(server_df): break
                        if server_df['time'][pos + add] == client_df['time 2'][index]:
                            add_to_df(index, pos, pos+add)
                            break




    #calculate the error
    error = []
    error_percentage = []
    
    # use this to get random sample (same sample size as constant speed, for comparison)
    comb_df = comb_df.sample(n=200, random_state=42)
    comb_df.reset_index(drop=True, inplace=True)

    # Error in meters
    # Percent error = |(actual - predicted)/actual|
    for index in range (len(comb_df)):
        error.append(abs(comb_df['dist_act'][index] - comb_df['dist_pred'][index]))
        error_percentage.append(abs((comb_df['dist_act'][index] - comb_df['dist_pred'][index])/comb_df['dist_act'][index]))

    # Print mean error and mean error %
    print(comb_df)
    print("number of samples: ", len(error))
    print("mean error in meters: ", sum(error)/len(error))
    print("mean error percentage: ",sum(error_percentage)/len(error_percentage))


if __name__ == '__main__':
    # Directories for csv files
    # Client csv from dtnet.py
    # Server csv from odometer.py
    directory = "/home/oem/Desktop/csv_files"
    client = "client_no.csv"
    server = "server_no.csv"
    clientcsv = os.path.join(directory, client)
    servercsv = os.path.join(directory, server)
    calc_dist(clientcsv, servercsv)