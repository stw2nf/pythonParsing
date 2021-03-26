import pandas
import math as mt
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy import signal

meter2ft = 3.28084
s2us = .000001

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

# %% Define location of csv message files

location =  r'C:\Users\Sean\Box\Descent Testing\flightRawData\flight67\flight67_'

# %% Local Position

#Find current message and save as string
currentMessage = r'vehicle_local_position_0.csv'
directory = location + currentMessage #concatenate location and message strings together

dfLocalPos = pandas.read_csv(directory)
localPos= dfLocalPos[['vx', 'vy']]
localPos = localPos.to_numpy() #Convert to array for ease of use down the line
vx = localPos[:,0]
vy = localPos[:,1]
vel = np.sqrt(vx**2 + vy**2)*meter2ft

tLocalPos = dfLocalPos[['timestamp']]
tLocalPos = tLocalPos.to_numpy() #Convert to array for ease of use down the line

# %% Power Consumption

currentMessage = r'battery_status_0.csv'
directory = location + currentMessage #concatenate location and message strings together

#Read in battery data 
dfPowerConsumption = pandas.read_csv(directory)

#Select relevant battery data
powerConsumption = dfPowerConsumption[['timestamp', 'voltage_filtered_v', 'current_filtered_a']]
tPowerConsumption = dfPowerConsumption[['timestamp']]
tPowerConsumption = tPowerConsumption.to_numpy() #Convert to array for ease of use down the line

voltage = powerConsumption['voltage_filtered_v'] #V
current = powerConsumption['current_filtered_a'] #A
pwrCons = voltage*current #Watts

# %% Plotting Altitude
fig1 = plt.figure()

ax1 = plt.subplot(2,1,1)
ax1.plot(tLocalPos, vel)
plt.xlabel('Time (us)', fontsize=8)
plt.ylabel('Speed (ft/s)')
plt.title('Translational Speed (ft/s)')

ax2 = plt.subplot(2,1,2, sharex=ax1)
ax2.plot(tPowerConsumption, current, label='Current (A)')
ax2.plot(tPowerConsumption, voltage, label='Voltage (V)')
ax2.plot(tPowerConsumption, pwrCons, label='Power Consumption (W)')
plt.xlabel('Time (us)', fontsize=8)
plt.title('Power Consumption')
ax2.legend()

plt.subplots_adjust(left=None, bottom=None, right=None, top=.9, wspace=None, hspace=.5)

plt.show()

#Prompt user for starting index of descent
print('Click Start of Translation Test')
start = plt.ginput(1, show_clicks=True)
startTimeStamp = start[0]
startTimeStamp = int(np.round(startTimeStamp[0])) #Need integer for indexing later on

#Prompt user for ending index of descent
print('Click End of Translation Test')
end = plt.ginput(1, show_clicks=True)
endTimeStamp = end[0]
endTimeStamp = int(round(endTimeStamp[0])) #Need integer for indexing later on

# %% Calculate Metrics for comparison from the Desecent
#Finding Local Position indices for the section of descent defined by the user
sLocalPos = np.where(tLocalPos == int(findClosest(tLocalPos, len(tLocalPos), startTimeStamp)))
eLocalPos = np.where(tLocalPos == int(findClosest(tLocalPos, len(tLocalPos), endTimeStamp)))
sLocalPos = sLocalPos[0][0]
eLocalPos = eLocalPos[0][0]

#Finding Power Consumption indices for the section of descent defined by the user
sPwr = np.where(tPowerConsumption == int(findClosest(tPowerConsumption, len(tPowerConsumption), startTimeStamp)))
ePwr = np.where(tPowerConsumption == int(findClosest(tPowerConsumption, len(tPowerConsumption), endTimeStamp)))
sPwr = sPwr[0][0]
ePwr = ePwr[0][0]

#Calculate mean Power Consumption over the course of the entire descent
pwrConsAverage = np.mean(pwrCons[sPwr:ePwr])
pwrConsStd = np.std(pwrCons[sPwr:ePwr])
velAverage = np.mean(vel[sLocalPos:eLocalPos])
velStd = np.std(vel[sLocalPos:eLocalPos])

results = np.atleast_2d(np.array([startTimeStamp, endTimeStamp, pwrConsAverage, pwrConsStd, velAverage, velStd]))