#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 20:15:33 2018

@author: david
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy import special
from scipy.optimize import minimize

epsilon = 0.00000001

# In[1]:
"""
-----------------------------------------------------------------------------------
In this cell, all functions for creating Poisson Regression model are provided.
-----------------------------------------------------------------------------------
"""

def Generate_Prob_of_Scores(H_goal, A_goal, Para_lambda, Para_mu):
    H_Poisson = np.power(Para_lambda,H_goal)*np.exp(-Para_lambda)/special.factorial(H_goal);
    A_Poisson = np.power(Para_mu,A_goal)*np.exp(-Para_mu)/special.factorial(A_goal)
    result = H_Poisson*A_Poisson
    return result

#end def

#H_G = np.array([1, 3, 2, 0])
#A_G = np.array([3, 2, 1, 1])
#lbda = np.array([1.2, 3.1, 2.1, 0.9])
#mu = np.array([1.0, 1.1, 0.6, 3.9])
#
#asvs = Generate_Prob_of_Scores(H_G, A_G, lbda, mu)
#kffw=H_G + asvs

def Compute_Time_Series_Likelihood(Input_Matrix, current_time):
    t_para = 0.0065
    time_eff_part = np.exp(-t_para*(current_time - Input_Matrix[:,0]))
    prob_part = -Input_Matrix[:,5]-Input_Matrix[:,6]+Input_Matrix[:,3]*np.log(
            Input_Matrix[:,5]+epsilon)+Input_Matrix[:,4]*np.log(Input_Matrix[:,6]+epsilon)
    result = np.sum(time_eff_part+prob_part)
    return result

#end def
    
def Predict_Prob_of_Match_Results(Para_lambda, Para_mu):
    Match_num = np.shape(Para_lambda)[0]
    Para_lambda = Para_lambda.reshape(Match_num)
    Para_mu = Para_mu.reshape(Match_num)
    H_Prob = np.zeros([Match_num])
    D_Prob = np.zeros([Match_num])
    A_Prob = np.zeros([Match_num])
    H_goal = np.zeros([Match_num])
    A_goal = np.zeros([Match_num])
    temp = np.zeros([Match_num])
    for i in np.arange(11):
        for j in np.arange(11):
            for k in np.arange(Match_num):
                H_goal[k] = i
                A_goal[k] = j
            if i>j:
                temp = Generate_Prob_of_Scores(i,j,Para_lambda,Para_mu)
                H_Prob+= temp
            elif i==j:
                temp = Generate_Prob_of_Scores(i,j,Para_lambda,Para_mu)
                D_Prob+= temp
            elif i<j:
                temp = Generate_Prob_of_Scores(i,j,Para_lambda,Para_mu)
                A_Prob+= temp
    H_Prob = np.reshape(H_Prob,(Match_num,1))
    D_Prob = np.reshape(D_Prob,(Match_num,1))
    A_Prob = np.reshape(A_Prob,(Match_num,1))
    result = np.concatenate((H_Prob,D_Prob,A_Prob),axis=1)
    return result

#end def
    
def Compute_Lambda(Input_Matrix,solution):
    Input_Matrix= Input_Matrix.astype(np.int32)
    train_data_size = np.shape(Input_Matrix)[0]
    sol_size = np.shape(solution)[0]
    
    attack_rate = solution[0:((sol_size-1)/2)]
    defend_rate = solution[((sol_size-1)/2):(sol_size-1)]
    Home_eff = solution[-1]
    
    temp_lambda = np.zeros([train_data_size,1])
    for i in np.arange(train_data_size):
        temp_lambda[i] = attack_rate[Input_Matrix[i,1]]*defend_rate[Input_Matrix[i,2]]*Home_eff
    result = temp_lambda
    return result

#end def
    
def Compute_Mu(Input_Matrix,solution):
    Input_Matrix= Input_Matrix.astype(np.int32)
    train_data_size = np.shape(Input_Matrix)[0]
    sol_size = np.shape(solution)[0]
    
    attack_rate = solution[0:((sol_size-1)/2)]
    defend_rate = solution[((sol_size-1)/2):(sol_size-1)]
    
    temp_mu = np.zeros([train_data_size,1])
    for i in np.arange(train_data_size):
        temp_mu[i] = attack_rate[Input_Matrix[i,2]]*defend_rate[Input_Matrix[i,1]]
    result = temp_mu
    return result

#end def

def Loss_Function_for_Train(solution):
    Initial_Input = Train_Data
    Initial_Input = Initial_Input.astype(np.int32)
    
    temp_lambda = Compute_Lambda(Initial_Input,solution)
    temp_mu = Compute_Mu(Initial_Input,solution)
    
    Final_Input = np.concatenate((Initial_Input,temp_lambda,temp_mu),axis=1)
    result = -Compute_Time_Series_Likelihood(Final_Input,T_current)
    return result

#end def
    
#Train_Data = np.array([[0,1,2,2,1],[1,3,5,3,1],[4,7,6,0,1]])
#T_current = 72
#
#input_sol = 1+np.random.random((17))
#
#test_loss = Loss_Function_for_Train(input_sol)

# In[2]:
"""
-----------------------------------------------------------------------------------
In this cell, the football match results are created.
-----------------------------------------------------------------------------------
"""
# England Primier League
epl_1516 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1516/E0.csv")
epl_1516 = epl_1516[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
epl_1516 = epl_1516.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
epl_1516 = epl_1516.dropna()

epl_1617 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1617/E0.csv")
epl_1617 = epl_1617[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
epl_1617 = epl_1617.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
epl_1617 = epl_1617.dropna()

epl_1718 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1718/E0.csv")
epl_1718 = epl_1718[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
epl_1718 = epl_1718.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
epl_1718 = epl_1718.dropna()

# England Championship
ech_1516 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1516/E1.csv")
ech_1516 = ech_1516[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
ech_1516 = ech_1516.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
ech_1516 = ech_1516.dropna()

ech_1617 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1617/E1.csv")
ech_1617 = ech_1617[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
ech_1617 = ech_1617.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
ech_1617 = ech_1617.dropna()

ech_1718 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1718/E1.csv")
ech_1718 = ech_1718[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
ech_1718 = ech_1718.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
ech_1718 = ech_1718.dropna()

# England League 1 
el1_1516 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1516/E2.csv")
el1_1516 = el1_1516[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
el1_1516 = el1_1516.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
el1_1516 = el1_1516.dropna()

el1_1617 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1617/E2.csv")
el1_1617 = el1_1617[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
el1_1617 = el1_1617.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
el1_1617 = el1_1617.dropna()

el1_1718 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1718/E2.csv")
el1_1718 = el1_1718[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
el1_1718 = el1_1718.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
el1_1718 = el1_1718.dropna()

# England League 2 
el2_1516 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1516/E3.csv")
el2_1516 = el2_1516[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
el2_1516 = el2_1516.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
el2_1516 = el2_1516.dropna()

el2_1617 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1617/E3.csv")
el2_1617 = el2_1617[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
el2_1617 = el2_1617.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
el2_1617 = el2_1617.dropna()

el2_1718 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1718/E3.csv")
el2_1718 = el2_1718[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
el2_1718 = el2_1718.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
el2_1718 = el2_1718.dropna()

# England Conference 
eco_1516 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1516/EC.csv")
eco_1516 = eco_1516[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
eco_1516 = eco_1516.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
eco_1516 = eco_1516.dropna()

eco_1617 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1617/EC.csv")
eco_1617 = eco_1617[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
eco_1617 = eco_1617.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
eco_1617 = eco_1617.dropna()

eco_1718 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1718/EC.csv")
eco_1718 = eco_1718[['Date','HomeTeam','AwayTeam','FTHG','FTAG','BbAvH','BbAvD','BbAvA']]
eco_1718 = eco_1718.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
eco_1718 = eco_1718.dropna()


All_Data = pd.concat([epl_1516,epl_1617,epl_1718,ech_1516,ech_1617,ech_1718,el1_1516,el1_1617,
                      el1_1718,el2_1516,el2_1617,el2_1718,eco_1516,eco_1617,eco_1718])

All_Data['Date_num'] = All_Data['Date'].apply(lambda x:
    time.mktime(time.strptime(x,'%d/%m/%y')))/3600
    
Data_size = All_Data.shape[0]
Feature_size = All_Data.shape[1]-1

Club_Name_Dict = {'Chelsea':0}
dict_num = 1
for i in np.arange(Data_size):
    if Club_Name_Dict.has_key(All_Data.iloc[i,1])==False:
        Club_Name_Dict[All_Data.iloc[i,1]]= dict_num
        dict_num +=1

Data_Num_Version = np.zeros([Data_size,Feature_size])

Data_Num_Version[:,0] = All_Data.iloc[:,Feature_size]
Data_Num_Version[:,3:Feature_size] = All_Data.iloc[:,3:Feature_size]

for i in np.arange(Data_size):
    Data_Num_Version[i,1] = Club_Name_Dict[All_Data.iloc[i,1]]
    Data_Num_Version[i,2] = Club_Name_Dict[All_Data.iloc[i,2]]

# In[3]:
"""
-----------------------------------------------------------------------------------
In this cell, the Poisson Regression model is trained, and the predictions
for match results are presented.
-----------------------------------------------------------------------------------
"""
time_sort_arg = np.argsort(Data_Num_Version[:,0])
Data_Time_Sorted = Data_Num_Version[time_sort_arg]
Train_Data_All = Data_Time_Sorted[:,:5]
Predict_Result = np.zeros([2,(Feature_size+5)])

team_num = len(Club_Name_Dict)

upper_bound_value = 100
temp_bound = ((0, upper_bound_value),(0, upper_bound_value))
temp_odd_bound = ((0, upper_bound_value),(0, upper_bound_value),(0, upper_bound_value))
x_bound = ((0, upper_bound_value),(0, upper_bound_value))
for i in np.arange((team_num-2)):
    x_bound += temp_bound
x_bound += temp_odd_bound

T_current = time.mktime(time.strptime(epl_1718.iloc[0,0],'%d/%m/%y'))/3600-24

while T_current<Data_Time_Sorted[-1,0]:
    print 'works well-------\n'
    sys.stdout.flush()
    temp_index = np.where(Train_Data_All[:,0] <= T_current)
    Train_Data = Train_Data_All[temp_index]
    temp_index = np.where(np.logical_and(Train_Data_All[:,0]>T_current, 
                                         Train_Data_All[:,0]<=T_current+24*14))
    Test_Data = Data_Time_Sorted[temp_index]
    if np.shape(Test_Data)[0]>0:
        x0 = np.random.random(2*team_num+1)
        fit_result = minimize(Loss_Function_for_Train, x0, method='L-BFGS-B', bounds=x_bound, tol=1e-4)
        temp_sol = fit_result.x
        temp_test_lambda = Compute_Lambda(Test_Data,temp_sol)
        temp_test_mu = Compute_Mu(Test_Data,temp_sol)
        temp_test_match_result = Predict_Prob_of_Match_Results(temp_test_lambda,temp_test_mu)
        Test_Data = np.concatenate((Test_Data,temp_test_lambda,temp_test_mu,
                                temp_test_match_result),axis=1)
        Predict_Result = np.concatenate((Predict_Result,Test_Data),axis=0)
    T_current+=24*14

Predict_Result = np.delete(Predict_Result, [0,1], axis=0)

np.savetxt('Prediction_Result_for_1718.csv', Predict_Result, delimiter = ',')










#Train_Data = np.array([[0,0,1,2,1],[1,2,1,3,1],[4,0,2,0,2]])
#T_current = 72
#x0 = np.random.random(7)
#x_bound = ((0, 2),(0, 2),(0, 2),(0, 2),(0, 2),(0, 2),(0, 2))
#res = minimize(Loss_Function_for_Train, x0, method='L-BFGS-B', bounds=x_bound, tol=1e-4)
#temp_sol = res.x
#               options={'gtol': 5e-2,'maxfun':10000})





















