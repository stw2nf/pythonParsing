#Importing ulog data and parsing

import pandas
import math as mt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math

meter2ft = 3.28084
s2us = .000001
rad2deg = 180/math.pi

# %% Quaternion to Euler conversion 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in deg (counterclockwise)
        pitch is rotation around y in deg (counterclockwise)
        yaw is rotation around z in deg (counterclockwise)
        """
        roll_x = np.zeros(len(x))
        pitch_y = np.zeros(len(x))
        yaw_z = np.zeros(len(x))
        
        for i in range(len(x)):
            t0 = +2.0 * (w[i] * x[i] + y[i] * z[i])
            t1 = +1.0 - 2.0 * (x[i] * x[i] + y[i] * y[i])
            roll_x[i] = np.arctan2(t1, t0)*rad2deg-90
         
            t2 = +2.0 * (w[i] * y[i] - z[i] * x[i])
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y[i] = math.asin(t2)*rad2deg
         
            t3 = +2.0 * (w[i] * z[i] + x[i] * y[i])
            t4 = +1.0 - 2.0 * (y[i] * y[i] + z[i] * z[i])
            yaw_z[i] = np.arctan2(t3, t4)*rad2deg
     
        return roll_x, pitch_y, yaw_z # in radians

# %% Functions
# Python3 program to find element 
# closet to given target. 

# Returns element closest to target in arr[] 
def findClosest(arr, n, target): 
    # Corner cases 
    if (target <= arr[0]): 
        return arr[0] 
    if (target >= arr[n - 1]): 
        return arr[n - 1] 
    # Doing binary search 
    i = 0; j = n; mid = 0
    while (i < j):  
        mid = int((i + j) / 2)
        if (arr[mid] == target): 
            return arr[mid] 
        # If target is less than array  
        # element, then search in left 
        if (target < arr[mid]) : 
            # If target is greater than previous 
            # to mid, return closest of two 
            if (mid > 0 and target > arr[mid - 1]): 
                return getClosest(arr[mid - 1], arr[mid], target) 
            # Repeat for left half  
            j = mid 
        # If target is greater than mid 
        else : 
            if (mid < n - 1 and target < arr[mid + 1]): 
                return getClosest(arr[mid], arr[mid + 1], target) 
            # update i 
            i = mid + 1  
    # Only single element left after search 
    return arr[mid] 
# Method to compare which one is the more close. 
# We find the closest by taking the difference 
# between the target and both values. It assumes 
# that val2 is greater than val1 and target lies 
# between these two. 
def getClosest(val1, val2, target): 
    if (target - val1 >= val2 - target): 
        return val2 
    else: 
        return val1 
# This code is contributed by Smitha Dinesh Semwal 

# %% Derivative Function
def deriv(arr, time):
    threshold=80
    darray = np.zeros(len(arr))
    for i in range(1,len(arr)):
        darray[i] = (arr[i]-arr[i-1])/(time[i]-time[i-1])
        if abs(darray[i]) > threshold:
            darray[i]=0
    return darray
# %% Moving Average Function
def movingAv(arr,n):
    arrAv = np.zeros(len(arr))
    for i in range(n,len(arr)):
        arrSum = 0
        for j in range(n):
            arrSum = arrSum + arr[i-j]
        arrAv[i] = arrSum/n
    return arrAv

# %% Define location of csv message files
files = ['48']
resultsAll =  np.zeros((len(files),6))

badPwrData = 50

for z in range(len(files)):
    
    eFlip = 0.331099362*60 #Expermentally derived energy required to flip (Watt*min)
    
    looksGood = 0
    
    location =  r'C:\Users\stw2nf\Box\Descent Testing\flightRawData\flight'+ files[z] + '\\flight' + files[z] + '_'
    
    # %% GPS and Local Position
    
    #Find current message and save as string
    currentMessage = r'vehicle_gps_position_0.csv'
    directory = location + currentMessage #concatenate location and message strings together
     
    #Read in GPS data for altitude (MSL) (m)
    df_GPS = pandas.read_csv(directory)
    altitude = (df_GPS[['alt']]/1000)*meter2ft
    altitude = altitude.to_numpy() #Convert to array for ease of use down the line
    tGPS = df_GPS[['timestamp']]
    tGPS = tGPS.to_numpy() #Convert to array for ease of use down the line
    
    dAltitude = deriv(altitude,(tGPS*s2us))
    ddAltitude = deriv(dAltitude,(tGPS*s2us))
    smoothDAlt = movingAv(dAltitude, 4)
    smoothDDAlt = movingAv(ddAltitude, 4)
    
    #Read in GPS data for local position (m) (Used later for Glide Ratio Estimation)
    currentMessage = r'vehicle_local_position_0.csv'
    directory = location + currentMessage #concatenate location and message strings together
    
    dfLocalPos = pandas.read_csv(directory)
    localPos= dfLocalPos[['x', 'y']]
    localPos = localPos.to_numpy() #Convert to array for ease of use down the line
    x_loc = localPos[:,0]
    y_loc = localPos[:,1]
    
    tLocalPos = dfLocalPos[['timestamp']]
    tLocalPos = tLocalPos.to_numpy() #Convert to array for ease of use down the line
    
    # %% Control State 
    
    #Find current message and save as string
    currentMessage = r'commander_state_0.csv'
    directory = location + currentMessage #concatenate location and message strings together
     
    #Read in GPS data for altitude (MSL) (m)
    df_ControlState = pandas.read_csv(directory)
    tControlState = df_ControlState[['timestamp']]
    tControlState = tControlState.to_numpy() #Convert to array for ease of use down the line
    
    mode = df_ControlState[['main_state']]
    mode = mode.to_numpy() #Convert to array for ease of use down the line
    
    # %% Vehicle Attitude Setpoints
    
    currentMessage = r'vehicle_attitude_setpoint_0.csv'
    directory = location + currentMessage #concatenate location and message strings together
    
    #Read in vehicle attitude setpoint data (Euler) 
    dfAttitudeSetpoint = pandas.read_csv(directory)
    
    #Select relevant Euler Setpoint data from dataframe (Radians)
    attitudeSetpoint = dfAttitudeSetpoint[['timestamp', 'roll_body', 'pitch_body', 'yaw_body']]
    tAttitudeSetpoint = dfAttitudeSetpoint[['timestamp']]
    tAttitudeSetpoint = tAttitudeSetpoint.to_numpy() #Convert to array for ease of use down the line
        
    #Convert to degrees and unwrap if necessary (Roll and Yaw)
    rollSetpoint = -np.array(np.rad2deg(attitudeSetpoint['roll_body']))
    pitchSetpoint = np.array(np.rad2deg(attitudeSetpoint['pitch_body']))
    yawSetpoint = np.array(np.rad2deg(attitudeSetpoint['yaw_body']))
    
    # %% Vehicle Attitude
    
    currentMessage = r'vehicle_attitude_0.csv'
    directory = location + currentMessage #concatenate location and message strings together
    
    #Read in vehicle attitude data (Quaternions) 
    dfAttitude = pandas.read_csv(directory)
    
    #Select relevant quaternion data from dataframe
    q0 = dfAttitude[['q[0]']].to_numpy()
    q1 = dfAttitude[['q[1]']].to_numpy()
    q2 = dfAttitude[['q[2]']].to_numpy()
    q3 = dfAttitude[['q[3]']].to_numpy()
    
    # q = dfAttitude[['q[1]', 'q[2]', 'q[3]','q[0]']]
    tAttitude = dfAttitude[['timestamp']]
    tAttitude = tAttitude.to_numpy() #Convert to array for ease of use down the line
    
    roll, pitch, yaw = euler_from_quaternion(q1, q2, q3, q0)
    rollUnwrapped = roll
    
    # %% Vehicle Rate
    
    currentMessage = r'vehicle_angular_velocity_0.csv'
    directory = location + currentMessage #concatenate location and message strings together
    
    #Read in vehicle attitude data (Quaternions) 
    dfRate = pandas.read_csv(directory)
    
    #Select relevant Euler Setpoint data from dataframe (Radians)
    rate = dfRate[['timestamp', 'xyz[0]', 'xyz[1]', 'xyz[2]']]
    tRate = dfRate[['timestamp']]
    tRate = tRate.to_numpy() #Convert to array for ease of use down the line
        
    #Convert to degrees and unwrap if necessary (Roll and Yaw)
    rollRate = -np.rad2deg(rate['xyz[0]'])
    pitchRate = np.rad2deg(rate['xyz[1]'])
    yawRate = np.rad2deg(rate['xyz[2]'])
    rollRate = rollRate.to_numpy() #Convert to array for ease of use down the line
    pitchRate = pitchRate.to_numpy() #Convert to array for ease of use down the line
    yawRate = yawRate.to_numpy() #Convert to array for ease of use down the line
    
    # %% Vehicle Rate Setpoints
    
    currentMessage = r'vehicle_rates_setpoint_0.csv'
    directory = location + currentMessage #concatenate location and message strings together
    
    #Read in vehicle attitude setpoint data (Euler) 
    dfRateSetpoint = pandas.read_csv(directory)
        
    #Select relevant Euler Setpoint data from dataframe (Radians)
    rateSetpoint = dfRateSetpoint[['timestamp', 'roll', 'pitch', 'yaw']]
    tRateSetpoint = dfRateSetpoint[['timestamp']]
    tRateSetpoint = tRateSetpoint.to_numpy() #Convert to array for ease of use down the line
        
    #Convert to degrees and unwrap if necessary (Roll and Yaw)
    rollRateSetpoint = -np.rad2deg(rateSetpoint['roll'])
    pitchRateSetpoint = np.rad2deg(rateSetpoint['pitch'])
    yawRateSetpoint = np.rad2deg(rateSetpoint['yaw'])
    rollRateSetpoint = rollRateSetpoint.to_numpy() #Convert to array for ease of use down the line
    pitchRateSetpoint = pitchRateSetpoint.to_numpy() #Convert to array for ease of use down the line
    yawRateSetpoint = np.roll(yawRateSetpoint.to_numpy() ,-180)#Convert to array for ease of use down the line
    
    # %% Power Consumption
    
    currentMessage = r'battery_status_0.csv'
    directory = location + currentMessage #concatenate location and message strings together
    
    #Read in battery data 
    dfPowerConsumption = pandas.read_csv(directory)
    
    #Select relevant battery data
    powerConsumption = dfPowerConsumption[['timestamp', 'voltage_filtered_v', 'current_filtered_a', 'discharged_mah']]
    tPowerConsumption = dfPowerConsumption[['timestamp']]
    tPowerConsumption = tPowerConsumption.to_numpy() #Convert to array for ease of use down the line
    
    voltage = powerConsumption['voltage_filtered_v'] #V
    current = powerConsumption['current_filtered_a'] #A
    pwrCons = voltage*current #Watts
    
    
    # %% Identify the descent
    
    #Finding where/if the vehicle switches to descent mode (Altitude or Invert)
    descentIndex = (np.where(mode != 2))        
    descentIndex = descentIndex[0]
    sDescent = int(tControlState[descentIndex[0]])
    eDescent = int(tControlState[descentIndex[-1]])
    
    startLag = int(sDescent+int(0.85/s2us)) #start descent plus .5 seconds of stabilization for inverted descent
    endEarly = int(eDescent-int(.6/s2us)) #Dont include flip in attitude/rate performance metrics
    
    if mode[descentIndex[0]]==1:
        eFlip=0
        sGPS = np.where(tGPS == int(findClosest(tGPS, len(tGPS), sDescent)))
        sGPS = sGPS[0][0]
        eGPS = np.where(tGPS == int(findClosest(tGPS, len(tGPS), eDescent)))
        eGPS = eGPS[0][0]
        for i in range(sGPS,eGPS):
            if (ddAltitude[i]>2.5):
                eDescent = tGPS[i]
                endEarly = eDescent
                break
    # %% Calculate Metrics for comparison from the Desecent
    
    #Finding GPS indices for the section of descent defined by the user
    sGPS = np.where(tGPS == int(findClosest(tGPS, len(tGPS), sDescent)))
    eGPS = np.where(tGPS == int(findClosest(tGPS, len(tGPS), eDescent)))
    sGPS = sGPS[0][0]
    eGPS = eGPS[0][0]
    
    #Finding Local Position indices for the section of descent defined by the user
    sLocalPos = np.where(tLocalPos == int(findClosest(tLocalPos, len(tLocalPos), sDescent)))
    eLocalPos = np.where(tLocalPos == int(findClosest(tLocalPos, len(tLocalPos), eDescent)))
    sLocalPos = sLocalPos[0][0]
    eLocalPos = eLocalPos[0][0]
    
    #Finding Attitude indices for the section of descent defined by the user
    sAtt = np.where(tAttitude == int(findClosest(tAttitude, len(tAttitude), startLag)))
    eAtt = np.where(tAttitude == int(findClosest(tAttitude, len(tAttitude), endEarly)))
    sAtt = sAtt[0][0]
    eAtt = eAtt[0][0]
    
    #Finding Attitude Setpoint indices for the section of descent defined by the user
    sAttSp = np.where(tAttitudeSetpoint == int(findClosest(tAttitudeSetpoint, len(tAttitudeSetpoint), startLag)))
    eAttSp = np.where(tAttitudeSetpoint == int(findClosest(tAttitudeSetpoint, len(tAttitudeSetpoint), endEarly)))
    sAttSp = sAttSp[0][0]
    eAttSp = eAttSp[0][0]
    
    #Finding Rate indices for the section of descent defined by the user
    sRate = np.where(tRate == int(findClosest(tRate, len(tRate), startLag)))
    eRate = np.where(tRate == int(findClosest(tRate, len(tRate), endEarly)))
    sRate = sRate[0][0]
    eRate = eRate[0][0]
    
    #Finding Rate Setpoint indices for the section of descent defined by the user
    sRateSp = np.where(tRateSetpoint == int(findClosest(tRateSetpoint, len(tRateSetpoint), startLag)))
    eRateSp = np.where(tRateSetpoint == int(findClosest(tRateSetpoint, len(tRateSetpoint), endEarly)))
    sRateSp = sRateSp[0][0]
    eRateSp = eRateSp[0][0]
    
    #Finding Power Consumption indices for the section of descent defined by the user (dont use fli)
    sPwr = np.where(tPowerConsumption == int(findClosest(tPowerConsumption, len(tPowerConsumption), startLag)))
    ePwr = np.where(tPowerConsumption == int(findClosest(tPowerConsumption, len(tPowerConsumption), endEarly)))
    sPwr = sPwr[0][0]
    ePwr = ePwr[0][0]
    sTotEn = np.where(tPowerConsumption == int(findClosest(tPowerConsumption, len(tPowerConsumption), sDescent)))
    eTotEn = np.where(tPowerConsumption == int(findClosest(tPowerConsumption, len(tPowerConsumption), endEarly)))
    sTotEn = sTotEn[0][0]
    eTotEn = eTotEn[0][0]
    #Calculate overall Desecent Rate over the course of the entire descent
    descentRateAverage = (altitude[eGPS] - altitude[sGPS]) / ((tGPS[eGPS] - tGPS[sGPS])*s2us)
    descentRate = np.zeros(eGPS-sGPS) 
    
    #Calculate instantaneous Desecent Rates over the course of the entire descent                      
    for k in (range(eGPS-sGPS)):
        descentRate[k]= (altitude[sGPS+k+1] - altitude[sGPS+k])/((tGPS[sGPS+k+1]-tGPS[sGPS+k])*s2us)
    descentRateAverage2 = np.mean(descentRate)
        
    #Calculate mean Power Consumption over the course of the entire descent
    pwrConsAverage = np.mean(pwrCons[sPwr:ePwr])        
    pwrConsStd = np.std(pwrCons[sPwr:ePwr])
    if pwrConsAverage < badPwrData:
        pwrConsAverage = 0
        pwrConsStd = 0
        
    totalEnergy = (float(pwrConsAverage*(tPowerConsumption[eTotEn]-tPowerConsumption[sTotEn])*s2us)*.017)+eFlip #Energy consumed in watt*
    if pwrConsAverage < badPwrData:
        totalEnergy = 0
        
    energyPerFoot = float(totalEnergy/(altitude[sGPS] - altitude[eGPS])) #watt*min/ft
    
    # #Calculate Mean Attitude Control Error
    
    #Resample Roll Angle / Setpoint if necessary
    rollSP_Window = rollSetpoint[sAttSp:eAttSp]
    rollAngle_Window = rollUnwrapped[sAtt:eAtt]
    
    if len(rollAngle_Window) > len(rollSP_Window):
        rollSP_Window = signal.resample(rollSP_Window, len(rollAngle_Window))
        
    else:
        rollAngle_Window = signal.resample(rollAngle_Window, len(rollSP_Window))
    
    #Resample Pitch Angle / Setpoint if necessary
    pitchSP_Window = pitchSetpoint[sAttSp:eAttSp]
    pitchAngle_Window = pitch[sAtt:eAtt]
    
    if len(pitchAngle_Window) > len(pitchSP_Window):
        pitchSP_Window = signal.resample(pitchSP_Window, len(pitchAngle_Window))
        
    else:
        pitchAngle_Window = signal.resample(pitchAngle_Window, len(pitchSP_Window))
    
    #Resample Yaw Angle / Setpoint if necessary
    yawSP_Window = yawSetpoint[sAttSp:eAttSp]
    yawAngle_Window = yaw[sAtt:eAtt]
    
    if len(yawAngle_Window) > len(yawSP_Window):
        yawSP_Window = signal.resample(yawSP_Window, len(yawAngle_Window))
        
    else:
        yawAngle_Window = signal.resample(yawAngle_Window, len(yawSP_Window))
        
    rollErrorMean = np.mean(abs(np.subtract(rollSP_Window, rollAngle_Window)))
    pitchErrorMean = np.mean(abs(np.subtract(pitchSP_Window, pitchAngle_Window)))
    yawErrorMean = np.mean(abs(np.subtract(yawSP_Window, yawAngle_Window)))
    
    #Calculate Standard Deviation of the Attitude Control Error
    rollErrorStd = np.std(abs(np.subtract(rollSP_Window, rollAngle_Window)))
    pitchErrorStd = np.std(abs(np.subtract(pitchSP_Window, pitchAngle_Window)))
    yawErrorStd = np.std(abs(np.subtract(yawSP_Window, yawAngle_Window)))
    
    # #Calculate Mean Rate Control Error
    
    #Resample Roll Rate / Setpoint if necessary
    rollRateSP_Window = rollRateSetpoint[sRateSp:eRateSp]
    rollRate_Window = rollRate[sRate:eRate]
    
    if len(rollRate_Window) > len(rollRateSP_Window):
        rollRateSP_Window = signal.resample(rollRateSP_Window, len(rollRate_Window))
        
    else:
        rollRate_Window = signal.resample(rollRate_Window, len(rollRateSP_Window))
        
    #Resample Pitch Rate / Setpoint if necessary
    pitchRateSP_Window = pitchRateSetpoint[sRateSp:eRateSp]
    pitchRate_Window = pitchRate[sRate:eRate]
    
    if len(pitchRate_Window) > len(pitchRateSP_Window):
        pitchRateSP_Window = signal.resample(pitchRateSP_Window, len(pitchRate_Window))
        
    else:
        pitchRate_Window = signal.resample(pitchRate_Window, len(pitchRateSP_Window))
    
    #Resample Yaw Rate / Setpoint if necessary
    yawRateSP_Window = yawRateSetpoint[sRateSp:eRateSp]
    yawRate_Window = yawRate[sRate:eRate]
    
    if len(yawRate_Window) > len(yawRateSP_Window):
        yawRateSP_Window = signal.resample(yawRateSP_Window, len(yawRate_Window))
        
    else:
        yawRate_Window = signal.resample(yawRate_Window, len(yawRateSP_Window))
        
    rollRateErrorMean = np.mean(abs(np.subtract(rollRateSP_Window, rollRate_Window)))
    pitchRateErrorMean = np.mean(abs(np.subtract(pitchRateSP_Window, pitchRate_Window)))
    yawRateErrorMean = np.mean(abs(np.subtract(yawRateSP_Window, yawRate_Window)))
    
    # #Calculate Standard Deviation of the Rate Control Error
    rollRateErrorStd = np.std(abs(np.subtract(rollRateSP_Window, rollRate_Window)))
    pitchRateErrorStd = np.std(abs(np.subtract(pitchRateSP_Window, pitchRate_Window)))
    yawRateErrorStd = np.std(abs(np.subtract(yawRateSP_Window, yawRate_Window)))
    
    #Calculate translation from XY local position
    dTranslated = (mt.hypot((x_loc[eLocalPos]-x_loc[sLocalPos]), (y_loc[eLocalPos]-y_loc[sLocalPos])))*meter2ft #Distance translated (ft)
    
    #Calculate Expermimental Glide Ratio
    glideRatio = dTranslated/(altitude[sGPS] - altitude[eGPS])
    
    #Create Array for all outputs
    results = np.atleast_2d(np.array([pwrConsAverage, energyPerFoot, float(glideRatio), rollErrorMean, pitchErrorMean, yawErrorMean]))
    resultsAll[z,:] = results
    
    # %% Plotting Altitude
    fig1 = plt.figure()
    
    ax1 = plt.subplot(3,1,1)
    ax1.plot(tGPS, altitude, label='Altitude (MSL) (ft)')
    plt.vlines(tGPS[sGPS], min(altitude), max(altitude), colors='k', linestyles='dashed')
    plt.vlines(tGPS[eGPS], min(altitude), max(altitude), colors='k', linestyles='dashed')
    plt.xlabel('Time (us)', fontsize=8)
    plt.ylabel('Altitude MSL (ft)')
    plt.title('Altitude of Entire Flight (MSL) (ft) Flight' + files[z])
    ax1.legend()
    
    ax2 = plt.subplot(3,1,2, sharex=ax1)
    ax2.plot(tPowerConsumption, current, label='Current (A)')
    ax2.plot(tPowerConsumption, voltage, label='Voltage (V)')
    ax2.plot(tPowerConsumption, pwrCons, label='Power Consumption (W)')
    plt.xlabel('Time (us)', fontsize=8)
    plt.title('Power Consumption')
    ax2.legend()
    
    ax3 = plt.subplot(3,1,3, sharex=ax1)
    ax3.plot(tAttitude, rollUnwrapped, label='Roll')
    ax3.plot(tAttitudeSetpoint, rollSetpoint, label='Roll Setpoint')
    ax3.plot(tAttitude, pitch, label='Pitch')
    ax3.plot(tAttitudeSetpoint, pitchSetpoint, label='Pitch Setpoint')
    ax3.plot(tAttitude, yaw, label='Yaw')
    ax3.plot(tAttitudeSetpoint, yawSetpoint, label='Yaw Setpoint')
    plt.xlabel('Time (us)', fontsize=8)
    plt.title('Attitude Control')
    ax3.legend()
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=.9, wspace=None, hspace=.5)
    
    plt.show()
    
    while looksGood == 0:
        looksGood = plt.ginput(1)
    
    plt.close()
    
    # # %% Plotting Attitude Control
    
    # fig2 = plt.figure()
    # ax = plt.subplot(111)
    
    # #Plot Attitude Estimates
    # ax.plot(tAttitude[sAtt:eAtt]*s2us, rollUnwrapped[sAtt:eAtt], label='Roll Angle')
    # ax.plot(tAttitude[sAtt:eAtt]*s2us, pitch[sAtt:eAtt], label='Pitch Angle')
    # ax.plot(tAttitude[sAtt:eAtt]*s2us, yaw[sAtt:eAtt], label='Yaw Angle')
    
    # #Plot Desired Attitude from Controller
    # ax.plot(tAttitudeSetpoint[sAttSp:eAttSp]*s2us, rollSetpoint[sAttSp:eAttSp], label='Roll Setpoint')
    # ax.plot(tAttitudeSetpoint[sAttSp:eAttSp]*s2us, pitchSetpoint[sAttSp:eAttSp], label='Pitch Setpoint')
    # ax.plot(tAttitudeSetpoint[sAttSp:eAttSp]*s2us, yawSetpoint[sAttSp:eAttSp], label='Yaw Setpoint')
    
    # plt.title('Control Input and Angle Response')
    # ax.legend()
    # plt.show()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Degrees')
    
    # # %% Plotting Rate Control
    
    # fig3 = plt.figure()
    # ax = plt.subplot(111)
    
    # #Plot Attitude Estimates
    # ax.plot(tRate[sRate:eRate]*s2us, rollRate[sRate:eRate], label='Roll Rate')
    # ax.plot(tRate[sRate:eRate]*s2us, pitchRate[sRate:eRate], label='Pitch Rate')
    # ax.plot(tRate[sRate:eRate]*s2us, yawRate[sRate:eRate], label='Yaw Rate')
    
    # #Plot Desired Attitude from Controller
    # ax.plot(tRateSetpoint[sRateSp:eRateSp]*s2us, rollRateSetpoint[sRateSp:eRateSp], label='Roll Rate Setpoint')
    # ax.plot(tRateSetpoint[sRateSp:eRateSp]*s2us, pitchRateSetpoint[sRateSp:eRateSp], label='Pitch Rate Setpoint')
    # ax.plot(tRateSetpoint[sRateSp:eRateSp]*s2us, yawRateSetpoint[sRateSp:eRateSp], label='Yaw Rate Setpoint')
    
    # plt.title('Control Input and Rate Response')
    # ax.legend()
    # plt.show()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Degrees/sec')
    
    # # %% Plotting Power Consumption
    # fig4 = plt.figure()
    # ax = plt.subplot(111)
    
    # #Plot Attitude Estimates
    # ax.plot(tPowerConsumption[sPwr:ePwr]*s2us, voltage[sPwr:ePwr], label='Voltage (V)')
    # ax.plot(tPowerConsumption[sPwr:ePwr]*s2us, current[sPwr:ePwr], label='Current (A)')
    # ax.plot(tPowerConsumption[sPwr:ePwr]*s2us, pwrCons[sPwr:ePwr], label='Power (W)')
    
    # plt.title('Power Consumption')
    # ax.legend()
    # plt.show()
    # plt.xlabel('Time (s)')
    
    # # %% Re - Plotting Altitude (with time as x axis)
    # fig6 = plt.figure()
    # ax = plt.subplot(111)
    
    # #Plot Attitude Estimates
    # ax.plot(tGPS[sGPS:eGPS]*s2us, altitude[sGPS:eGPS], label='Altitude (MSL) (ft)')
    
    # plt.title('Altitude of Descent (MSL) (ft)')
    # ax.legend()
    # plt.show()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Altitude MSL (ft)')
    
    # # %% Plotting Descent Rate (with time as x axis)
    # fig7 = plt.figure()
    # ax = plt.subplot(111)
    
    # #Plot Attitude Estimates
    # ax.plot(tGPS[sGPS:eGPS]*s2us, descentRate, label='Descent Rate (ft/sec)')
    
    # plt.title('Descent Rate (ft/sec)')
    # ax.legend()
    # plt.show()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Descent Rate (ft/sec)')
    
    # # %% Plotting Descent Rate (with time as x axis)
    # fig8 = plt.figure()
    # ax = plt.subplot(111)
    
    # ax.plot(tGPS[sGPS:eGPS]*s2us, dAltitude[sGPS:eGPS], label='Altitude Derivative (ft/sec)')
    # ax.plot(tGPS[sGPS:eGPS]*s2us, smoothDAlt[sGPS:eGPS], label='Smoothed Derivative (ft/sec)')
    # ax.plot(tGPS[sGPS:eGPS]*s2us, ddAltitude[sGPS:eGPS], label='Altitude Alpha (ft/sec)')
    # ax.plot(tGPS[sGPS:eGPS]*s2us, smoothDDAlt[sGPS:eGPS], label='Smoothed Alpha (ft/sec)')
    
    # plt.title('Descent Rate (ft/sec)')
    # ax.legend()
    # plt.show()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Descent Rate (ft/sec)')
    
    # # %% Plotting Power with flip lag (with time as x axis)
    # fig9 = plt.figure()
    # ax = plt.subplot(111)
    
    # #Plot Attitude Estimates
    # ax.plot(tPowerConsumption[sTotEn:eTotEn]*s2us, pwrCons[sTotEn:eTotEn], label='Power (W)')
    
    # plt.title('Power Consumption')
    # ax.legend()
    # plt.show()
    # plt.xlabel('Time (s)')
