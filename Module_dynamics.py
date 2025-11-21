#Module containing classes and functions regarding dynamics of solar powered electric vehicles
#2025-10-14 Merlin Tétrault-Leclerc
#_________________________________________________________________________________
#_________________________________________________________________________________


#LIBRARIES
#_________________________________________________________________________________
import os

import csv

import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import brute
from scipy.integrate import solve_ivp

import pandas as pd

import pytz as pytz

import timezonefinder as tzf

import pvlib
from pvlib.location import Location

from tqdm import tqdm

# CLASSES DEFINITION
#_________________________________________________________________________________
class Path():
    #type is either from point A to point B (A2B) or is a lap (A2A)
    #TMY_resolution in long/lat degrees (approx 25km PVGIS spacial resolution for weather data)
    #Horizon_resolution in long/lat degrees (3 arcseconds PVGIS spacial resolution for elevation data)
    #Creates folder structure in accordance with the initialization folder structure
    #Expects a .txt file with the same name as the Path.name containing the lat, long, altitude of each point in the path
    def __init__(self, name, type = 'A2A'):
        self.TMY_resolution = 0.25  
        self.Horizon_resolution = 3/3600 
        self.name = str(name)
        self.type = str(type)

        #create the folder structure
        if not os.path.exists(self.name):
            os.makedirs(self.name)
        if not os.path.exists(self.name + '/TMY'):
            os.makedirs(self.name + '/TMY')
        if not os.path.exists(self.name + '/Horizon'):
            os.makedirs(self.name + '/Horizon')
    
    #generates the mapdata file from the .txt file with the same Path.name name
    #iterations is the number of times to filter the mapdata based on slope
    #the mapdata file is saved in the Path.name folder as mapdata.csv
    #the mapdata file contains the following columns:
    #lat, long, altitude, Dlat, Dlong, Daltitude, position, distance, direction, Dposition, slope, TMY_GridID, Horizon_GridID, timezone
    def gen_mapdata(self, iterations = 1, slope_threshold = 0.4):
        mapdata = np.loadtxt(self.name + '.txt', dtype=float)
        
        for loop in range(iterations):
            DlatArray = np.gradient(mapdata[:,0])

            DlongArray = np.gradient(mapdata[:,1])

            DAltArray = np.gradient(mapdata[:,2])

            PosArray = np.zeros(len(mapdata))
            for i in range(len(mapdata)):
                if i == 0:
                    PosArray[i] = 0
                else:
                    PosArray[i] = haversine(mapdata[i-1,0], mapdata[i-1,1], mapdata[i,0], mapdata[i,1])+PosArray[i-1]

            DPosArray = np.gradient(PosArray[:])

            DistArray = np.zeros(len(mapdata))
            for i in range(len(PosArray)):
                if i == 0:
                    DistArray[i] = 0
                else:
                    DistArray[i] = np.sqrt((PosArray[i]-PosArray[i-1])**2+(mapdata[i,2]-mapdata[i-1,2])**2) + DistArray[i-1]

            SlopeArray = np.arctan(DAltArray[:]/DPosArray[:])
            for i in range(len(SlopeArray)):
                if np.isnan(SlopeArray[i]):
                    SlopeArray[i] = 0

            #0rad is north and positive clockwise, pi/2 is east, pi is south, -pi/2 is west
            DirectionArray = np.arctan2(DlongArray, DlatArray)
            for i in range(len(DirectionArray)):
                if np.isnan(DirectionArray[i]):
                    DirectionArray[i] = 0

            if loop == iterations-1:
                continue
            mask = (np.abs(SlopeArray) <= slope_threshold)
            mask[0] = True  # Keep the first point
            mapdata = mapdata[mask]

        TimeZoneArray = []
        tf = tzf.TimezoneFinder()
        for i in range(len(mapdata)):
            lat = mapdata[i,0]
            long = mapdata[i,1]
            tz = tf.timezone_at(lat=lat, lng=long)
            TimeZoneArray.append(tz)
        TimeZoneArray = np.array(TimeZoneArray).reshape(-1, 1)

        #loop through mapdata and create TMY/Horizon grid ID for each point
        TMY_GridIDArray = []
        Horizon_GridIDArray = []
        for i in range(len(mapdata)):
            lat = float(mapdata[i, 0])
            long = float(mapdata[i, 1])
            TMY_latID = int(np.sign(lat)*(np.abs(np.trunc(lat/self.TMY_resolution))+1))
            TMY_longID = int(np.sign(long)*(np.abs(np.trunc(long/self.TMY_resolution))+1))
            TMY_GridID = str(TMY_latID) + ';' + str(TMY_longID)
            Horizon_latID = int(np.sign(lat)*(np.abs(np.trunc(lat/self.Horizon_resolution))+1))
            Horizon_longID = int(np.sign(long)*(np.abs(np.trunc(long/self.Horizon_resolution))+1))
            Horizon_GridID = str(Horizon_latID) + ';' + str(Horizon_longID)

            TMY_GridIDArray.append(TMY_GridID)
            Horizon_GridIDArray.append(Horizon_GridID)
            
        mapdata = np.column_stack((mapdata, DlongArray))            #3
        mapdata = np.column_stack((mapdata, DlatArray))             #4
        mapdata = np.column_stack((mapdata, DAltArray))             #5
        mapdata = np.column_stack((mapdata, PosArray))              #6
        mapdata = np.column_stack((mapdata, DistArray))             #7
        mapdata = np.column_stack((mapdata, DirectionArray))        #8
        mapdata = np.column_stack((mapdata, DPosArray))             #9 
        mapdata = np.column_stack((mapdata, SlopeArray))            #10
        mapdata = np.column_stack((mapdata, TMY_GridIDArray))       #11
        mapdata = np.column_stack((mapdata, Horizon_GridIDArray))   #12
        mapdata = np.column_stack((mapdata, TimeZoneArray))         #13

        mapdata = pd.DataFrame(mapdata, columns=['lat', 'long', 'altitude', 'Dlat', 'Dlong', 'Daltitude', 'position', 'distance', 'direction', 'Dposition', 'slope', 'TMY_GridID', 'Horizon_GridID', 'timezone'])

        # Convert relevant columns to numeric types
        numeric_columns = ['lat', 'long', 'altitude', 'Dlat', 'Dlong', 'Daltitude', 'position', 'distance', 'direction', 'Dposition', 'slope']
        mapdata[numeric_columns] = mapdata[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
        #save mapdata in the PathName folder
        mapdata.to_csv(self.name + '/mapdata.csv', index=False)

    #generates a mapdata file from specified slope, lat and long
    #It keeps the same fictive lat, long coordinates for each points in the path, altitude points are created based on the slope
    #the mapdata file is saved in the Path.name folder as mapdata.csv
    #the mapdata file contains the following columns:
    #lat, long, altitude, Dlat, Dlong, Daltitude, position, distance, direction, Dposition, slope, timezone
    def gen_mapdata_manual(self, slope, lat, long):
        mapdata = np.array([
        [lat, long],
        [lat, long]
        ], dtype=float)

        AltArray = np.zeros(len(mapdata))
        for i in range(len(mapdata)):
            if i == 0:
                AltArray[i] = 0
            else:
                AltArray[i] = AltArray[i-1] + 100*np.tan(np.deg2rad(slope))
        mapdata = np.column_stack((mapdata, AltArray))

        DlatArray = np.gradient(mapdata[:,0])

        DlongArray = np.gradient(mapdata[:,1])

        DAltArray = np.gradient(mapdata[:,2])

        PosArray = np.zeros(len(mapdata))
        for i in range(len(mapdata)):
            if i == 0:
                PosArray[i] = 0
            else:
                PosArray[i] = PosArray[i-1] + 100

        DPosArray = np.gradient(PosArray[:])

        DistArray = np.zeros(len(mapdata))
        for i in range(len(PosArray)):
            if i == 0:
                DistArray[i] = 0
            else:
                DistArray[i] = np.sqrt((PosArray[i]-PosArray[i-1])**2+(mapdata[i,2]-mapdata[i-1,2])**2) + DistArray[i-1]

        SlopeArray = np.arctan(DAltArray[:]/DPosArray[:])
        for i in range(len(SlopeArray)):
            if np.isnan(SlopeArray[i]):
                SlopeArray[i] = 0

        #0rad is north and positive clockwise, pi/2 is east, pi is south, -pi/2 is west
        DirectionArray = np.arctan2(DlongArray, DlatArray)
        for i in range(len(DirectionArray)):
            if np.isnan(DirectionArray[i]):
                DirectionArray[i] = 0

        TimeZoneArray = []
        tf = tzf.TimezoneFinder()
        for i in range(len(mapdata)):
            lat = mapdata[i,0]
            long = mapdata[i,1]
            tz = tf.timezone_at(lat=lat, lng=long)
            TimeZoneArray.append(tz)
        TimeZoneArray = np.array(TimeZoneArray).reshape(-1, 1)

        #loop through mapdata and create TMY/Horizon grid ID for each point
        TMY_GridIDArray = []
        Horizon_GridIDArray = []
        for i in range(len(mapdata)):
            lat = float(mapdata[i, 0])
            long = float(mapdata[i, 1])
            TMY_latID = int(np.sign(lat)*(np.abs(np.trunc(lat/self.TMY_resolution))+1))
            TMY_longID = int(np.sign(long)*(np.abs(np.trunc(long/self.TMY_resolution))+1))
            TMY_GridID = str(TMY_latID) + ';' + str(TMY_longID)
            Horizon_latID = int(np.sign(lat)*(np.abs(np.trunc(lat/self.Horizon_resolution))+1))
            Horizon_longID = int(np.sign(long)*(np.abs(np.trunc(long/self.Horizon_resolution))+1))
            Horizon_GridID = str(Horizon_latID) + ';' + str(Horizon_longID)

            TMY_GridIDArray.append(TMY_GridID)
            Horizon_GridIDArray.append(Horizon_GridID)
            
        mapdata = np.column_stack((mapdata, DlongArray))            #3
        mapdata = np.column_stack((mapdata, DlatArray))             #4
        mapdata = np.column_stack((mapdata, DAltArray))             #5
        mapdata = np.column_stack((mapdata, PosArray))              #6
        mapdata = np.column_stack((mapdata, DistArray))             #7
        mapdata = np.column_stack((mapdata, DirectionArray))        #8
        mapdata = np.column_stack((mapdata, DPosArray))             #9 
        mapdata = np.column_stack((mapdata, SlopeArray))            #10
        mapdata = np.column_stack((mapdata, TMY_GridIDArray))       #11
        mapdata = np.column_stack((mapdata, Horizon_GridIDArray))   #12
        mapdata = np.column_stack((mapdata, TimeZoneArray))         #13

        mapdata = pd.DataFrame(mapdata, columns=['lat', 'long', 'altitude', 'Dlat', 'Dlong', 'Daltitude', 'position', 'distance', 'direction', 'Dposition', 'slope', 'TMY_GridID', 'Horizon_GridID', 'timezone'])

        # Convert relevant columns to numeric types
        numeric_columns = ['lat', 'long', 'altitude', 'Dlat', 'Dlong', 'Daltitude', 'position', 'distance', 'direction', 'Dposition', 'slope']
        mapdata[numeric_columns] = mapdata[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
        #save mapdata in the PathName folder
        mapdata.to_csv(self.name + '/mapdata.csv', index=False)

    #loads the mapdata file as a pandas DataFrame
    #the mapdata file is expected to be in the Path.name folder as mapdata.csv
    #raises FileNotFoundError if the mapdata file is not found
    def get_mapdata(self):
        if os.path.exists(self.name + '/mapdata.csv'):
            mapdata = pd.read_csv(self.name + '/mapdata.csv')
            return mapdata
        else:
            raise FileNotFoundError(f"Map data file {self.name}/mapdata.csv not found")

    #generates the TMY files for each TMY_GridID in the mapdata file with the PVGIS API through pvlib
    #find and fills gaps in TMY grid IDs
    #the TMY files are saved in the Path.name/TMY folder as GridID_TMY.csv
    def gen_TMY(self):
        mapdata = self.get_mapdata()
        TMY_GridIDs = mapdata['TMY_GridID'].unique()

        #first pass to create TMY files for all unique TMY_GridIDs
        for TMY_GridID in TMY_GridIDs:
            filename = f"{self.name}/TMY/{TMY_GridID}_TMY.csv"
            lat = float(TMY_GridID.split(';')[0])*self.TMY_resolution + .0001
            long = float(TMY_GridID.split(';')[1])*self.TMY_resolution + .0001
            if not os.path.exists(filename):
                CurrentTMYData, metadata = pvlib.iotools.get_pvgis_tmy(lat, long, 
                                                        outputformat='json',
                                                        usehorizon=False,
                                                        userhorizon=None, 
                                                        startyear=None, 
                                                        endyear=None, 
                                                        map_variables=True, 
                                                        url=pvlib.iotools.pvgis.URL, 
                                                        timeout=30, 
                                                        roll_utc_offset=None, 
                                                        coerce_year=None)
                CurrentTMYData['Month_Day_Time'] = pd.to_datetime(CurrentTMYData.index.strftime('2025-%m-%d %H:%M:%S')).tz_localize('UTC')
                CurrentTMYData['Month_Day_Time_numeric'] = CurrentTMYData['Month_Day_Time'].apply(lambda x: x.timestamp())
                CurrentTMYData.to_csv(filename, index=True)

        #second pass to fill gaps in TMY grid IDs
        TMY_GridID = TMY_GridIDs[0]
        current_TMY_Grid_latID = int(TMY_GridID.split(';')[0])
        current_TMY_Grid_longID = int(TMY_GridID.split(';')[1])
        for TMY_GridID in TMY_GridIDs:
            TMY_latID, TMY_longID = map(int, TMY_GridID.split(';'))
            Delta_LatID = TMY_latID - current_TMY_Grid_latID
            Delta_LongID = TMY_longID - current_TMY_Grid_longID
            lat = float(TMY_GridID.split(';')[0])*self.TMY_resolution
            long = float(TMY_GridID.split(';')[1])*self.TMY_resolution
            if abs(Delta_LatID) == abs(Delta_LongID):
                for i in range(abs(Delta_LatID)):
                    filename = f"{self.name}/TMY/{str(TMY_latID - i*np.sign(Delta_LatID))};{str(TMY_longID - (i-1)*np.sign(Delta_LatID))}_TMY.csv"
                    if not os.path.exists(filename):
                        CurrentTMYData, metadata = pvlib.iotools.get_pvgis_tmy(lat, long, 
                                                                outputformat='json',
                                                                usehorizon=False,
                                                                userhorizon=None, 
                                                                startyear=None, 
                                                                endyear=None, 
                                                                map_variables=True, 
                                                                url=pvlib.iotools.pvgis.URL, 
                                                                timeout=30, 
                                                                roll_utc_offset=None, 
                                                                coerce_year=None)
                        CurrentTMYData['Month_Day_Time'] = pd.to_datetime(CurrentTMYData.index.strftime('2025-%m-%d %H:%M:%S')).tz_localize('UTC')
                        CurrentTMYData['Month_Day_Time_numeric'] = CurrentTMYData['Month_Day_Time'].apply(lambda x: x.timestamp())
                        CurrentTMYData.to_csv(filename, index=True)
                    filename = f"{self.name}/TMY/{str(TMY_latID - i*np.sign(Delta_LatID))};{str(TMY_longID - i*np.sign(Delta_LongID))}_TMY.csv"
                    if not os.path.exists(filename):
                        CurrentTMYData, metadata = pvlib.iotools.get_pvgis_tmy(lat, long, 
                                                                outputformat='json',
                                                                usehorizon=False,
                                                                userhorizon=None, 
                                                                startyear=None, 
                                                                endyear=None, 
                                                                map_variables=True, 
                                                                url=pvlib.iotools.pvgis.URL, 
                                                                timeout=30, 
                                                                roll_utc_offset=None, 
                                                                coerce_year=None)
                        CurrentTMYData['Month_Day_Time'] = pd.to_datetime(CurrentTMYData.index.strftime('2025-%m-%d %H:%M:%S')).tz_localize('UTC')
                        CurrentTMYData['Month_Day_Time_numeric'] = CurrentTMYData['Month_Day_Time'].apply(lambda x: x.timestamp())
                        CurrentTMYData.to_csv(filename, index=True)
                    filename = f"{self.name}/TMY/{str(TMY_latID - (i-1)*np.sign(Delta_LatID))};{str(TMY_longID - i*np.sign(Delta_LongID))}_TMY.csv"
                    if not os.path.exists(filename):
                        CurrentTMYData, metadata = pvlib.iotools.get_pvgis_tmy(lat, long, 
                                                                outputformat='json',
                                                                usehorizon=False,
                                                                userhorizon=None, 
                                                                startyear=None, 
                                                                endyear=None, 
                                                                map_variables=True, 
                                                                url=pvlib.iotools.pvgis.URL, 
                                                                timeout=30, 
                                                                roll_utc_offset=None, 
                                                                coerce_year=None)
                        CurrentTMYData['Month_Day_Time'] = pd.to_datetime(CurrentTMYData.index.strftime('2025-%m-%d %H:%M:%S')).tz_localize('UTC')
                        CurrentTMYData['Month_Day_Time_numeric'] = CurrentTMYData['Month_Day_Time'].apply(lambda x: x.timestamp())
                        CurrentTMYData.to_csv(filename, index=True)
            elif abs(Delta_LatID) + abs(Delta_LongID) > 2:
                for i in range(abs(Delta_LatID)+1):
                    for j in range(abs(Delta_LongID)+1):
                        filename = f"{self.name}/TMY/{str(TMY_latID - i*np.sign(Delta_LatID))};{str(TMY_longID - j*np.sign(Delta_LongID))}_TMY.csv"
                        if not os.path.exists(filename):
                            CurrentTMYData, metadata = pvlib.iotools.get_pvgis_tmy(lat, long, 
                                                                    outputformat='json',
                                                                    usehorizon=False,
                                                                    userhorizon=None, 
                                                                    startyear=None, 
                                                                    endyear=None, 
                                                                    map_variables=True, 
                                                                    url=pvlib.iotools.pvgis.URL, 
                                                                    timeout=30, 
                                                                    roll_utc_offset=None, 
                                                                    coerce_year=None)
                            CurrentTMYData['Month_Day_Time'] = pd.to_datetime(CurrentTMYData.index.strftime('2025-%m-%d %H:%M:%S')).tz_localize('UTC')
                            CurrentTMYData['Month_Day_Time_numeric'] = CurrentTMYData['Month_Day_Time'].apply(lambda x: x.timestamp())
                            CurrentTMYData.to_csv(filename, index=True)
            current_TMY_Grid_latID = TMY_latID
            current_TMY_Grid_longID = TMY_longID      

    #loads the TMY file as a pandas DataFrame
    #the TMY file is expected to be in the Path.name/TMY folder as GridID_TMY.csv
    #raises FileNotFoundError if the TMY file is not found
    def get_TMY(self, TMY_GridID):
        TMY_filename = f"{self.name}/TMY/{TMY_GridID}_TMY.csv"
        if os.path.exists(TMY_filename):
            TMY_data = pd.read_csv(TMY_filename)
            return TMY_data
        else:
            raise FileNotFoundError(f"TMY file {TMY_filename} not found")
        
    #generates the Horizon files for each Horizon_GridID in the mapdata file with the PVGIS API through pvlib
    #find and fills gaps in Horizon grid IDs
    #the Horizon files are saved in the Path.name/Horizon folder as GridID_Horizon.csv
    def gen_Horizon(self):
        mapdata = self.get_mapdata()
        Horizon_GridIDs = mapdata['Horizon_GridID'].unique()

        #first pass to create Horizon files for all unique Horizon_GridIDs
        for Horizon_GridID in Horizon_GridIDs:
            filename = f"{self.name}/Horizon/{Horizon_GridID}_Horizon.csv"
            lat = float(Horizon_GridID.split(';')[0])*self.Horizon_resolution
            long = float(Horizon_GridID.split(';')[1])*self.Horizon_resolution
            if not os.path.exists(filename):
                CurrentHorizonData, metadata = pvlib.iotools.get_pvgis_horizon(lat, long, 
                                                            url=pvlib.iotools.pvgis.URL)
                CurrentHorizonData.to_csv(filename, index=True)                    

        #second pass to fill gaps in TMY grid IDs
        Horizon_GridID = Horizon_GridIDs[0]
        current_Horizon_Grid_latID = int(Horizon_GridID.split(';')[0])
        current_Horizon_Grid_longID = int(Horizon_GridID.split(';')[1])
        for Horizon_GridID in Horizon_GridIDs:
            Horizon_latID, Horizon_longID = map(int, Horizon_GridID.split(';'))
            Delta_LatID = Horizon_latID - current_Horizon_Grid_latID
            Delta_LongID = Horizon_longID - current_Horizon_Grid_longID
            lat = float(Horizon_GridID.split(';')[0])*self.Horizon_resolution + .0001
            long = float(Horizon_GridID.split(';')[1])*self.Horizon_resolution + .0001
            if abs(Delta_LatID) == abs(Delta_LongID):
                for i in range(abs(Delta_LatID)):
                    filename = f"{self.name}/Horizon/{str(Horizon_latID - i*np.sign(Delta_LatID))};{str(Horizon_longID - (i-1)*np.sign(Delta_LongID))}_Horizon.csv"
                    if not os.path.exists(filename):
                        CurrentHorizonData, metadata = pvlib.iotools.get_pvgis_horizon(lat, long, 
                                                            url=pvlib.iotools.pvgis.URL)
                        CurrentHorizonData.to_csv(filename, index=True)
                    filename = f"{self.name}/Horizon/{str(Horizon_latID - i*np.sign(Delta_LatID))};{str(Horizon_longID - i*np.sign(Delta_LongID))}_Horizon.csv"
                    if not os.path.exists(filename):
                        CurrentHorizonData, metadata = pvlib.iotools.get_pvgis_horizon(lat, long, 
                                                            url=pvlib.iotools.pvgis.URL)
                        CurrentHorizonData.to_csv(filename, index=True)
                    filename = f"{self.name}/Horizon/{str(Horizon_latID - (i-1)*np.sign(Delta_LatID))};{str(Horizon_longID - i*np.sign(Delta_LongID))}_Horizon.csv"
                    if not os.path.exists(filename):
                        CurrentHorizonData, metadata = pvlib.iotools.get_pvgis_horizon(lat, long, 
                                                            url=pvlib.iotools.pvgis.URL)
                        CurrentHorizonData.to_csv(filename, index=True)
            elif abs(Delta_LatID) + abs(Delta_LongID) > 1:
                for i in range(abs(Delta_LatID)+1):
                    for j in range(abs(Delta_LongID)+1):
                        filename = f"{self.name}/Horizon/{str(Horizon_latID - i*np.sign(Delta_LatID))};{str(Horizon_longID - j*np.sign(Delta_LongID))}_Horizon.csv"
                        if not os.path.exists(filename):
                            CurrentHorizonData, metadata = pvlib.iotools.get_pvgis_horizon(lat, long, 
                                                            url=pvlib.iotools.pvgis.URL)
                            CurrentHorizonData.to_csv(filename, index=True) 
            current_Horizon_Grid_latID = Horizon_latID
            current_Horizon_Grid_longID = Horizon_longID    

    #loads the Horizon file as a pandas DataFrame
    #the Horizon file is expected to be in the Path.name/Horizon folder as GridID_Horizon.csv
    #raises FileNotFoundError if the TMY file is not found
    def get_Horizon(self, Horizon_GridID):
        Horizon_filename = f"{self.name}/Horizon/{Horizon_GridID}_Horizon.csv"
        if os.path.exists(Horizon_filename):
            Horizon_data = pd.read_csv(Horizon_filename)
            return Horizon_data
        else:
            raise FileNotFoundError(f"Horizon file {Horizon_filename} not found")
        
class Environment():
    #Environment class to store environment constants, such as air density and gravity
    #Also have functions to retrieve environment variables, such as
    #Floor variables : slope, altitude
    def __init__(self, path:Path, StartDateTimeLocal = '2025-06-15 08:00:00'):
        self.rho = 1.2 # kg/m^3, air density at sea level and 20 degrees Celsius
        self.g = 9.81 # m/s^2, acceleration due to gravity
        self.path = path
        self.mapdata = self.path.get_mapdata()
        self.lapdistance = max(self.mapdata['distance'])
        self.lapcount = 0
        self.slope = None
        self.direction = None
        self.location = Location(latitude=self.mapdata.loc[0, 'lat'], longitude=self.mapdata.loc[0, 'long'], tz=self.mapdata.loc[0, 'timezone'], altitude=self.mapdata.loc[0, 'altitude'])
        self.TMY_GridID = None
        self.TMYData = None
        self.Horizon_GridID = None
        self.HorizonData = None
        self.StartDateTimeLocal = pd.to_datetime(StartDateTimeLocal).tz_localize(self.mapdata.loc[0,'timezone'])
        self.StartDateTimeUTC = self.StartDateTimeLocal.tz_convert('UTC')
        self.DateTimeUTC = None
        self.DateTimeLocal = None
        self.pathcomplete = False
        self.weather = None
        self.solarposition = None
        self.day = None
        self.shade = None

        distance_array = self.mapdata['distance'].values
        lat_array = self.mapdata['lat'].values
        long_array = self.mapdata['long'].values
        altitude_array = self.mapdata['altitude'].values
        slope_array = self.mapdata['slope'].values
        direction_array = self.mapdata['direction'].values
        timezone_array = self.mapdata['timezone'].values
        self.InterpArrays = [distance_array, lat_array, long_array, altitude_array, slope_array, direction_array, timezone_array]
    
    #Checks if a lap has been completed and update the lapcount attribute
    #Returns a True/False PathComplete to use with a logic check to end simulation before the simulation time
    def CheckLap(self, Distance, PathType):
        if Distance > self.lapdistance*(self.lapcount+1):
            if PathType == 'A2B':
                self.pathcomplete = True
            elif PathType == 'A2A':
                self.lapcount = self.lapcount + 1
                self.pathcomplete = False

    #Updates the location attribute and the grid IDs attributes
    #Updates the slope and direction attributes
    def get_location(self, Distance, TMY_res, Horizon_res):
        current_timezone = self.InterpArrays[6][np.argmax(self.InterpArrays[0] >= (Distance - self.lapcount * self.lapdistance))]
        lat = np.interp((Distance-self.lapcount*self.lapdistance), self.InterpArrays[0], self.InterpArrays[1])
        long = np.interp((Distance-self.lapcount*self.lapdistance), self.InterpArrays[0], self.InterpArrays[2])
        altitude = np.interp((Distance-self.lapcount*self.lapdistance), self.InterpArrays[0], self.InterpArrays[3])
        self.slope = np.interp((Distance-self.lapcount*self.lapdistance), self.InterpArrays[0], self.InterpArrays[4])
        self.direction = np.interp((Distance-self.lapcount*self.lapdistance), self.InterpArrays[0], self.InterpArrays[5])
        self.location = Location(latitude=lat, longitude=long, tz = current_timezone, altitude=altitude) #pvlib location object

        TMY_latID = int(np.sign(lat)*(np.abs(np.trunc(lat/TMY_res))+1))
        TMY_longID = int(np.sign(long)*(np.abs(np.trunc(long/TMY_res))+1))
        self.TMY_GridID = str(TMY_latID) + ';' + str(TMY_longID)

        Horizon_latID = int(np.sign(lat)*(np.abs(np.trunc(lat/Horizon_res))+1))
        Horizon_longID = int(np.sign(long)*(np.abs(np.trunc(long/Horizon_res))+1))
        self.Horizon_GridID = str(Horizon_latID) + ';' + str(Horizon_longID)

    #updates the weather attribute 1-D list: WindDirection, WindSpeed, Pressure, Temperature, dni, ghi, dhi
    #updates the attribute SolarPosition data structure: zenith, apparent zenith, height, apparent height and azimut
    #updates attributes day = True or False and shade = True or False
    def get_weather(self, Time, TMYData, HorizonData):
        self.weather = CurrentWeather(TMYData, Time)
        self.solarposition = self.location.get_solarposition(Time, pressure=self.weather[2], temperature = self.weather[3], method='nrel_numpy')
        
        HorizonHeight_interp = interp1d(HorizonData['horizon_azimuth'], 
                                        HorizonData['horizon_elevation'],
                                        kind='linear', 
                                        fill_value="extrapolate")
        HorizonHeight = HorizonHeight_interp(self.solarposition['azimuth'].values[0])
        if self.solarposition['zenith'].values[0] > 90:
            self.day = False
        else:
            self.day = True
            if self.solarposition['zenith'].values[0] > 90 - HorizonHeight:
                self.shade = True
            else:
                self.shade = False

class Rider():
    #Rider class to model the rider of the vehicle
    #In case of a human powered vehicle, the attribute Rider.power is the power output of the rider
    #The Rider.shift attribute is a list of tuples containing the start and end time of each riding/driving shift in local time (24h format)
    #The isriding method checks if the current local time is within a shift and updates the Rider.riding attribute (True/False)
    def __init__(self, mass = 70, power = 100, shifts = [(8, 12), (13, 17)]):
        self.mass = float(mass)  # kg
        self.power = float(power)  # W
        self.shifts = shifts
        self.riding = None
    
    def isriding(self, LocalTime):
        hour = LocalTime.hour + LocalTime.minute/60 + LocalTime.second/3600
        self.riding = any(start <= hour < end for start, end in self.shifts)
    
class Battery():
    #Battery class to model the battery of the vehicle
    #capacity is in Wh, SOC_Llim and SOC_Ulim are the lower and upper limits of the state of charge (0-1)
    #Crate_charge and Crate_discharge are the C-rates for charging and discharging
    #the battery_power method calculates the power that can be delivered by the battery based on the state of charge and the requested power
    def __init__(self, capacity, SOC_Llim = 0.1, SOC_Ulim = 0.95, Crate_charge = 1, Crate_discharge = 1):
        self.capacity = float(capacity) # Wh
        self.Crate_charge = float(Crate_charge) # C-rate for charging 
        self.Crate_discharge = float(Crate_discharge) # C-rate for discharging
        self.SOC_Ulim = float(SOC_Ulim)  # maximum state of charge (0-1)
        self.SOC_Llim = float(SOC_Llim)  # minimum state of charge (0-1)
        self.SOC = None

    #method to calculate a power factor based on the SOC of the battery, used in the Fp_motor method of the Vehicle class
    def battery_Pf(self):
        Pf_bat = (self.SOC)/(self.SOC_Ulim - self.SOC_Llim)-self.SOC_Llim/(self.SOC_Ulim - self.SOC_Llim)
        if Pf_bat > 1:
            Pf_bat = 1
        elif Pf_bat < 0:
            Pf_bat = 0
        else:
            Pf_bat = Pf_bat
        return Pf_bat
    
    #Calculates the power that can be delivered by the battery based on the SOC and the requested power
    def battery_power(self, Power):
        capacity = self.capacity * 3600  # convert Wh to Ws (J)
        battery_power = Power  # default: allow requested power
        if Power > 0:
            if Power > self.Crate_charge * capacity:
                battery_power = self.Crate_charge * capacity
        elif Power < 0:
            if Power < -self.Crate_discharge * capacity:
                battery_power = -self.Crate_discharge * capacity
        else:
            battery_power = 0
        return battery_power

    #not used currently, attribute SOC is updated directly in the dynamics method of the Vehicle class
    def new_SOC(self, delta_E):
        capacity = self.capacity * 3600  # convert Wh to Ws (J)
        if capacity == 0:
            self.SOC = 0
        Energy = self.SOC * capacity  # convert SOC to Ws (J)
        Energy = Energy + delta_E
        if Energy > capacity:
            Energy = capacity
        elif Energy < 0:
            Energy = 0
        else:
            Energy = Energy
        if capacity == 0:
            self.SOC = 0
        else:
            self.SOC = Energy/capacity

class SolarPanel():
    #SolarPanel class to model the solar panel of the vehicle
    #area is in m^2, efficiency is the efficiency of the solar panel (0-1)
    #the solar_power method calculates the power that can be delivered by the solar panel based on the irradiance and the area
    def __init__(self, area, efficiency = 0.2):
        self.area = float(area)  # m^2
        self.efficiency = float(efficiency)  # dimensionless (solar panel efficiency)
        self.I = None #plane of array irradiance at specified weather
        self.solarpower = None #amount of watts produced at I and weather

    #Calculates the plane of array irradiance for a fixed horizontal solar array
    #CurrentWeather, SolarPostition, day, shade are the exact outputs from Environment.Weather
    #slope and direction are outputs from Vehicule.CurrentLocation
    def get_FixedSolarPower_isotropic(self, Weather, SolarPostition, day, shade, slope, direction):
        if day == True:
            SurfaceTilt = np.rad2deg(slope)
            if slope < 0:
                SurfaceAzimuth = np.rad2deg(direction)
            elif slope >= 0:
                SurfaceAzimuth = -1*np.rad2deg(direction)
            else:
                SurfaceAzimuth = 180 #does not affect plane of array irradiance since tilt is 0

            IrradianceData = pvlib.irradiance.get_total_irradiance(surface_tilt=SurfaceTilt,
                                                    surface_azimuth=SurfaceAzimuth,
                                                    solar_zenith=SolarPostition['zenith'].values[0],
                                                    solar_azimuth=SolarPostition['azimuth'].values[0],
                                                    dni=Weather[4],
                                                    ghi=Weather[5],
                                                    dhi=Weather[6],
                                                    albedo = 0.25,
                                                    model='isotropic')

            if shade == True:
                self.I = IrradianceData['poa_diffuse']
            else:
                self.I = IrradianceData['poa_global']
        else:
            self.I = 0
        self.solarpower = self.I * self.area*self.efficiency

class Chassis():
    #Chassis class to store constants such as drag and rolling ressitance coeffs and mass
    #cargo_mass is added transported mass
    def __init__(self, mass, CdA, Crr):
        self.mass = float(mass)  # kg
        self.CdA = float(CdA)    # m^2
        self.Crr = float(Crr)    # dimensionless (rolling resistance coefficient)
        self.cargo_mass = 0

class Motor():
    #Motor class to model the motor of the vehicle
    #RatedPower is the max allowable power that can be used for propulsion
    #The motor_Pf method calculates the allowable power output based on the vehicle speed
    def __init__(self, RatedPower = 250, efficiency = 1):
        self.RatedPower = RatedPower
        self.efficiency = efficiency
        self.CutoffSpeed = 40/3.6 # m/s
        self.MaxRegenSpeed = 25 # m/s


    def motor_Pf(self, v):
        Pf_motor = -0.007*v**2 + 1
        if Pf_motor > 1:
            Pf_motor = 1
        elif Pf_motor < 0:
            if v < 0:
                Pf_motor = 1
            elif v >= 2*np.sqrt(1/.007):
                Pf_motor = -1
            else:
                Pf_motor = 0.007*v**2 - 0.33466*v + 3
        return Pf_motor

class Vehicle():
    #Vehicle class to model the vehicle dynamics assembling all other classes
    #The dynamics method is used with a numerical integrator to solve the equations of motion
    #
    def __init__(self, environment:Environment, rider:Rider, battery:Battery, solarpanel:SolarPanel, chassis:Chassis, motor:Motor):
        self.environment = environment
        self.rider = rider
        self.battery = battery
        self.solarpanel = solarpanel
        self.chassis = chassis
        self.motor = motor
        self.mass = None
        self.relativewindspeed = None
        self.motorpower = None
        self.mass = self.chassis.mass + self.rider.mass + self.chassis.cargo_mass
        self.environment_filter_alpha = 0.1

    def get_totalmass(self):
        chassis = self.chassis.mass
        cargo = self.chassis.cargo_mass
        rider = self.rider.mass
        self.mass = chassis+cargo+rider

    def Fr_rolling(self, v, slope):
        Fr_rolling = self.mass*self.environment.g*self.chassis.Crr*np.cos(slope)*-np.sign(v) if v != 0 else 0
        return Fr_rolling

    def Fr_drag(self, v, wind_v):
        Fr_drag = 0.5*self.environment.rho*self.chassis.CdA*(v - wind_v)**2*-np.sign(v - wind_v) if v != 0 else 0
        return Fr_drag
    
    def Fr_gravity(self, v,  slope):
        Fr_gravity = -1*self.mass*self.environment.g*np.sin(slope) if v != 0 else 0
        return Fr_gravity
    
    def Fp_rider(self,v):
        Fp_rider = self.rider.power/v if v != 0 else 0
        return Fp_rider
    
    def Fp_motor(self, v):
        Pf_bat = self.battery.battery_Pf()
        Pf_motor = self.motor.motor_Pf(v)
        if Pf_motor > Pf_bat:
            Pf_motor = Pf_bat
        elif Pf_bat == 1:
            Pf_motor = 1
        
        P_motor = self.battery.battery_power(Pf_motor*self.motor.RatedPower)
        if v != 0:
            Fp_motor = P_motor/v 
        else:
            P_motor = 0
            Fp_motor = 0
        return Fp_motor, P_motor
    
    def F_tot(self, v, slope, wind_v):
        Fr_rolling = self.Fr_rolling(v, slope)
        Fr_drag = self.Fr_drag(v, wind_v)
        Fr_gravity = self.Fr_gravity(v,slope)
        Fp_rider = self.Fp_rider(v)
        Fp_motor,P_motor= self.Fp_motor(v)
        F_tot = Fr_rolling + Fr_drag + Fr_gravity + Fp_rider + Fp_motor
        return F_tot, P_motor

    def F_tot_brake(self, v, slope, wind_v):
        Fr_rolling = self.Fr_rolling(v, slope)
        Fr_drag = self.Fr_drag(v, wind_v)
        Fr_gravity = self.Fr_gravity(v,slope)
        Fp_motor,P_motor= self.Fp_motor(self.motor.MaxRegenSpeed)
        F_tot = Fr_rolling + Fr_drag + Fr_gravity + Fp_motor
        return F_tot, P_motor

    def dynamics(self, t, X):
        x,v,E = X
    
        E = np.clip(E, 0, self.battery.capacity * 3600)
        if self.battery.capacity == 0:
            self.battery.SOC = 0
        else:
            self.battery.SOC = E/(self.battery.capacity * 3600)

        self.environment.DateTimeUTC = self.environment.StartDateTimeUTC + pd.Timedelta(seconds=t)
        self.environment.DateTimeLocal = self.environment.DateTimeUTC.tz_convert(self.environment.location.tz)

        self.environment.CheckLap(x, self.environment.path.type)

        Previous_TMY_GridID = self.environment.TMY_GridID
        Previous_Horizon_GridID = self.environment.Horizon_GridID
        self.environment.get_location(x, self.environment.path.TMY_resolution, self.environment.path.Horizon_resolution)

        if (self.environment.TMY_GridID != Previous_TMY_GridID) or (self.environment.TMYData is None):
            self.environment.TMYData = self.environment.path.get_TMY(self.environment.TMY_GridID)
        if (self.environment.Horizon_GridID != Previous_Horizon_GridID) or (self.environment.HorizonData is None):
            self.environment.HorizonData = self.environment.path.get_Horizon(self.environment.Horizon_GridID)

        self.environment.get_weather(self.environment.DateTimeUTC, self.environment.TMYData, self.environment.HorizonData)
        self.rider.isriding(self.environment.DateTimeLocal)


        if hasattr(self, 'prev_slope') == False:
            self.prev_slope = self.environment.slope if self.environment.slope is not None else 0
            self.prev_weather_0 = self.environment.weather[0] if self.environment.weather is not None else 0
            self.prev_weather_1 = self.environment.weather[1] if self.environment.weather is not None else 0
            
        #Smooth environmental variables to avoid numerical instabilities
        self.environment.slope = (1 - self.environment_filter_alpha) * self.prev_slope + self.environment_filter_alpha * self.environment.slope
        self.environment.weather = list(self.environment.weather)
        self.environment.weather[0] = (1 - self.environment_filter_alpha) * self.prev_weather_0 + self.environment_filter_alpha * self.environment.weather[0]
        self.environment.weather[1] = (1 - self.environment_filter_alpha) * self.prev_weather_1 + self.environment_filter_alpha * self.environment.weather[1] 

        self.prev_slope = self.environment.slope if self.environment.slope is not None else 0
        self.prev_weather_0 = self.environment.weather[0] if self.environment.weather is not None else 0
        self.prev_weather_1 = self.environment.weather[1] if self.environment.weather is not None else 0

        self.solarpanel.get_FixedSolarPower_isotropic(self.environment.weather, self.environment.solarposition, self.environment.day, self.environment.shade, self.environment.slope, self.environment.direction)

        if self.rider.riding == True:

            if self.environment.weather[0] > 180:
                WindDirection = self.environment.weather[0] - 360
            WindDirection = self.environment.weather[0] - 180
            WindDirection = np.deg2rad(WindDirection)
            self.relativewindspeed = self.environment.weather[1] * np.cos(WindDirection - self.environment.direction)

            net_force, P_motor = self.F_tot(v, self.environment.slope, self.relativewindspeed)
        else:
            if self.environment.weather[0] > 180:
                WindDirection = self.environment.weather[0] - 360
            WindDirection = self.environment.weather[0] - 180
            WindDirection = np.deg2rad(WindDirection)
            self.relativewindspeed = self.environment.weather[1] * np.cos(WindDirection - self.environment.direction)

            net_force, P_motor = self.F_tot_brake(v, self.environment.slope, self.relativewindspeed)

            if v <= 0:
                v = 0
                net_force = 0
                P_motor = 0

        a = net_force / self.mass

        P = self.solarpanel.solarpower - P_motor

        # If E is at lower limit and P would decrease it, set P=0 (no more discharge)
        if E <= 0 and P < 0:
            P = 0
        # If E is at upper limit and P would increase it, set P=0 (no more charge)
        if E >= self.battery.capacity * 3600 and P > 0:
            P = 0

        self.motorpower = P_motor

        return [v, a, P]

# FUNCTIONS DEFINITION
#_________________________________________________________________________________

#calculates the distance between two lat/long coordinates points accounting for the curvature of the earth
def haversine(lat1, long1, lat2, long2):
    Earth_radius = 6371000 #m
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    long1 = np.radians(long1)
    long2 = np.radians(long2)
    dist = 2*Earth_radius*np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((long2-long1)/2)**2))
    return dist

#gets current conditions at time time [UTC] in a TMY Data structure
def CurrentWeather(TMYData, Time):
    Time_numeric = Time.timestamp()
    WindDirection_interp = interp1d(TMYData['Month_Day_Time_numeric'], 
                                TMYData['wind_direction'], 
                                kind='linear', 
                                fill_value="extrapolate")
    WindSpeed_interp = interp1d(TMYData['Month_Day_Time_numeric'], 
                                TMYData['wind_speed'], 
                                kind='linear', 
                                fill_value="extrapolate")
    Pressure_interp = interp1d(TMYData['Month_Day_Time_numeric'], 
                                    TMYData['pressure'], 
                                    kind='linear', 
                                    fill_value="extrapolate")
    Temperature_interp = interp1d(TMYData['Month_Day_Time_numeric'], 
                                TMYData['temp_air'], 
                                kind='linear', 
                                fill_value="extrapolate")
    dni_interp = interp1d(TMYData['Month_Day_Time_numeric'], 
                                    TMYData['dni'], 
                                    kind='linear', 
                                    fill_value="extrapolate")
    ghi_interp = interp1d(TMYData['Month_Day_Time_numeric'], 
                                    TMYData['ghi'], 
                                    kind='linear', 
                                    fill_value="extrapolate")
    dhi_interp = interp1d(TMYData['Month_Day_Time_numeric'], 
                                    TMYData['dhi'], 
                                    kind='linear', 
                                    fill_value="extrapolate")

    WindDirection = WindDirection_interp(Time_numeric)
    WindSpeed = WindSpeed_interp(Time_numeric)
    WindSpeed = pvlib.atmosphere.windspeed_powerlaw(WindSpeed,10,1.5, exponent = None, surface_type = 'unstable_air_above_human_inhabited_areas')
    Pressure = Pressure_interp(Time_numeric)
    Temperature = Temperature_interp(Time_numeric)
    dni = dni_interp(Time_numeric)
    ghi = ghi_interp(Time_numeric)
    dhi = dhi_interp(Time_numeric)
    return WindDirection, WindSpeed, Pressure, Temperature, dni, ghi, dhi

#different simulate functions
#solves vehicle.dynamics for a specified simulation time, dt and initial SOC 
#initial position and velocity are set to 0 and are then controlled by the rider shifts conditions
def simulate_fixedtimestep(vehicle:Vehicle, simtime_s = 60*60*1, dt =1 , SOC_i = 1, output = "full"):
    mass = vehicle.mass
    motor_power = vehicle.motor.RatedPower
    solar_area = vehicle.solarpanel.area
    battery_capacity = vehicle.battery.capacity
    path = vehicle.environment.path.name
    os.makedirs('SIMRESULTS', exist_ok=True)
    filename = f"SIMRESULTS/mass_{mass}_motor_{motor_power}_solar_{solar_area}_battery_{battery_capacity}_{path}.csv"

    # Skip simulation if file exists
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping simulation.")
        return

    x_i = 0  # m
    v_i = 0 / 3.6  # m/s
    vehicle.battery.SOC = SOC_i
    E_i = vehicle.battery.capacity * 3600 * vehicle.battery.SOC  # Ws

    X = np.array([x_i, v_i, E_i])

    simtimes = np.arange(0, simtime_s, dt)
    nsteps = 0

    # Define CSV header
    if output == "full":
        header = [
            "simtime", "Distance", "Velocity", "Energy", "Times_local", "Times_UTC",
            "lat", "long", "altitude", "timezone", "slope", "direction",
            "WindDirection", "WindSpeed", "Pressure", "Temperature",
            "dni", "ghi", "dhi", "Relativewindspeed", "Motorpower",
            "POAIrradiance", "Solarpower"
        ]
    elif output == "light":
        header = [
            "simtime", "Distance", "Velocity", "Energy", "Times_local", "Times_UTC",
            "lat", "long", 
        ]
    else:
        raise ValueError("output must be 'full' or 'light'")

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for t in tqdm(simtimes, desc= "simulating"):
            # Exit if path is complete
            if getattr(vehicle.environment, "pathcomplete", False):
                break

            if vehicle.rider.riding == True:
                if X[1] < 0:
                    print('negative velocity at time', t, 'distance', X[0], 'speed', X[1])
                    vehicle.environment.pathcomplete = True
                elif X[1] < 0.2:
                    X[1] = 0.2

            if vehicle.rider.riding == False and X[1] < 0:
                X[1] = 0  # set velocity to 0 if not riding
   
            dX = np.array(vehicle.dynamics(t, X)) * dt
            X = X + dX

            if output == "full":
                # Extract location and weather attributes
                loc = vehicle.environment.location
                weather = vehicle.environment.weather

                row = [
                    t,
                    X[0],  # Distance
                    X[1],  # Velocity
                    X[2],  # Energy
                    vehicle.environment.DateTimeLocal,
                    vehicle.environment.DateTimeUTC,
                    loc.latitude,
                    loc.longitude,
                    loc.altitude,
                    str(loc.tz),
                    vehicle.environment.slope,
                    vehicle.environment.direction,
                    weather[0],  # WindDirection
                    weather[1],  # WindSpeed
                    weather[2],  # Pressure
                    weather[3],  # Temperature
                    weather[4],  # dni
                    weather[5],  # ghi
                    weather[6],  # dhi
                    vehicle.relativewindspeed,
                    vehicle.motorpower,
                    vehicle.solarpanel.I,
                    vehicle.solarpanel.solarpower
                ]
            else: #light
                # Extract location only
                loc = vehicle.environment.location

                row = [
                    t,
                    X[0],  # Distance
                    X[1],  # Velocity
                    X[2],  # Energy
                    vehicle.environment.DateTimeLocal,
                    vehicle.environment.DateTimeUTC,
                    loc.latitude,
                    loc.longitude,
                ]
            writer.writerow(row)

def simulate_RK45(vehicle, simtime_s = 60*60*1, SOC_i = 1):
    x_i = 0 #m
    v_i = 0/3.6 #m/s
    vehicle.battery.SOC = SOC_i
    E_i = vehicle.battery.capacity*3600*vehicle.battery.SOC #Ws

    res = solve_ivp(vehicle.dynamics, [0, simtime_s], [x_i, v_i, E_i], method='RK45', t_eval=np.arange(0, simtime_s, 1),  first_step = 0.1, rtol = 1e-4, atol = 2e-7)

    # Slice simtimes and all result arrays to the actual simulation length
    simtimes = res.t
    Xs = res.y

        # Build DataFrame
    df = pd.DataFrame({
        "simtime": simtimes,
        "Distance": Xs[0, :],
        "Velocity": Xs[1, :],
        "Energy": Xs[2, :],
    })
    mass = vehicle.mass
    motor_power = vehicle.motor.RatedPower
    solar_area = vehicle.solarpanel.area
    battery_capacity = vehicle.battery.capacity
    path = vehicle.environment.path.name
    os.makedirs('SIMRESULTS', exist_ok=True)
    filename = f"SIMRESULTS/mass_{mass}_motor_{motor_power}_solar_{solar_area}_battery_{battery_capacity}_{path}.csv"
    df.to_csv(filename, index=False)

def simulate_variabletimestep(vehicle:Vehicle, simtime_s = 60*60*1, atol=[1e-3, 1e-3, 1.0], rtol=1e-3, 
                              base_dt = 1, max_dt=2, max_dv = 0.2, SOC_i = 1, output = "full", use_pbar=True,
                              skip_if_result_exist = True):
    mass = vehicle.mass
    motor_power = vehicle.motor.RatedPower
    solar_area = vehicle.solarpanel.area
    battery_capacity = vehicle.battery.capacity
    path = vehicle.environment.path.name
    os.makedirs('SIMRESULTS', exist_ok=True)
    filename = f"SIMRESULTS/mass_{mass}_motor_{motor_power}_solar_{solar_area}_battery_{battery_capacity}_{path}.csv"

    # Skip simulation if file exists
    if skip_if_result_exist and os.path.exists(filename):
        print(f"File {filename} already exists. Skipping simulation.")
        return

    x_i = 0  # m
    v_i = 0 / 3.6  # m/s
    vehicle.battery.SOC = SOC_i
    E_i = vehicle.battery.capacity * 3600 * vehicle.battery.SOC  # Ws

    X = np.array([x_i, v_i, E_i])
    t = 0.0
    t_factor = 1.0

    atol = np.array(atol)  # exemple: [m, m/s, J]

    end_time = simtime_s
    max_dt_user = max_dt 

    # Define CSV header
    if output == "full":
        header = [
            "simtime", "Distance", "Velocity", "Energy", "Times_local", "Times_UTC",
            "lat", "long", "altitude", "timezone", "slope", "direction",
            "WindDirection", "WindSpeed", "Pressure", "Temperature",
            "dni", "ghi", "dhi", "Relativewindspeed", "Motorpower",
            "POAIrradiance", "Solarpower"
        ]
    elif output == "light":
        header = [
            "simtime", "Distance", "Velocity", "Energy", "Times_local", "Times_UTC",
            "lat", "long", 
        ]
    else:
        raise ValueError("output must be 'full' or 'light'")


    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
    
        if use_pbar: 
            pbar = tqdm(total=int(end_time), desc="simulating")

        try:
            while t < end_time:
                accept_step = True
                X_prev = X.copy()

                #Spetial logic when resting
                if vehicle.rider.riding == True:
                    if X[1] < 0:
                        print('negative velocity at time', t, 'distance', X[0], 'speed', X[1])
                        break
                    elif X[1] < 0.2:
                        X[1] = 0.2
                    max_dt = max_dt_user
                else:
                    max_dt = max_dt_user*10

                if vehicle.rider.riding == False and X[1] < 0:
                    X[1] = 0  # set velocity to 0 if not riding
                
                # Runge-Kutta 4(5) method
                dt = base_dt * t_factor
                X_1_dot = np.array(vehicle.dynamics(t, X))  # slope at 0
                X_2_dot = np.array(vehicle.dynamics(t + dt/2, X + X_1_dot*dt/2))  # RK2 term
                X_3_dot = np.array(vehicle.dynamics(t + dt/2, X + X_2_dot*dt/2))  # RK3 term
                X_4_dot = np.array(vehicle.dynamics(t + dt, X + X_3_dot*dt))  # RK4 term

                dX = (dt/6)*(X_1_dot + 2*X_2_dot + 2*X_3_dot + X_4_dot)
                dX_minor = (dt/6)*(X_1_dot + 4*X_2_dot + X_3_dot)
                X = X + dX
                scale = atol + rtol * np.maximum(np.abs(X_prev), np.abs(X_prev + dX))
                err_vec = np.abs(dX - dX_minor) / scale
                err = np.max(err_vec)
                
                #Adaptive time step control

                #Error based adaptive time step control
                safety = 0.9
                p = 3 # ordre du schéma "minor"
                if err == 0:
                    err_factor = 2.0  #Maximum dt factor if no error
                else:
                    err_factor = safety * (1.0 / err)**(1.0 / (p + 1))

                #Discontinuity based adaptive time step control
                if dX[1] == 0: 
                    discontinuity_factor = 2.0 #Maximum dt factor if no error
                else:
                    discontinuity_factor = safety * (max_dv / abs(dX[1]))**(1.0 / (1 + 1))

                #Adapt time step, only one factor is applied, the most restrictive
                factor = min(err_factor, discontinuity_factor)
                t_factor *= factor
                

                #Finals adjustements to t_factor
                if base_dt*t_factor > max_dt:
                    t_factor = max_dt/base_dt

                #Accept or reject step
                if abs(dX[1]) > max_dv:
                    accept_step = False

                if accept_step == False:
                    X = X_prev # Revert state
                else:
                    t += dt # Advance time

                if use_pbar:
                    # Manually control bar position = simulated time
                    pbar.n = min(int(t), int(end_time))   # clamp just in case
                    pbar.refresh()              # redraw

                    # (optional) show diagnostics on the bar
                    pbar.set_postfix(err=f"{err:.2e}", dt=f"{dt:.4f}")

                if accept_step:
                    if output == "full":
                        # Extract location and weather attributes
                        loc = vehicle.environment.location
                        weather = vehicle.environment.weather

                        row = [
                            t,
                            X[0],  # Distance
                            X[1],  # Velocity
                            X[2],  # Energy
                            vehicle.environment.DateTimeLocal,
                            vehicle.environment.DateTimeUTC,
                            loc.latitude,
                            loc.longitude,
                            loc.altitude,
                            str(loc.tz),
                            vehicle.environment.slope,
                            vehicle.environment.direction,
                            weather[0],  # WindDirection
                            weather[1],  # WindSpeed
                            weather[2],  # Pressure
                            weather[3],  # Temperature
                            weather[4],  # dni
                            weather[5],  # ghi
                            weather[6],  # dhi
                            vehicle.relativewindspeed,
                            vehicle.motorpower,
                            vehicle.solarpanel.I,
                            vehicle.solarpanel.solarpower
                        ]
                    else: #light
                        # Extract location only
                        loc = vehicle.environment.location

                        row = [
                            t,
                            X[0],  # Distance
                            X[1],  # Velocity
                            X[2],  # Energy
                            vehicle.environment.DateTimeLocal,
                            vehicle.environment.DateTimeUTC,
                            loc.latitude,
                            loc.longitude,
                        ]
                    writer.writerow(row)


        finally:
            # Make sure bar ends in a clean state
            if t < end_time and getattr(vehicle.environment, "pathcomplete", False):
                # If you want it to look 'finished' even when ending early:
                pbar.n = end_time
                pbar.refresh()
            pbar.close()
