# rFpro (2023b) auto generated test script [29/07/2024 10:58:43]

import clr # pythonnet
import os.path
import time

# Load the rFpro.Controller DLL/Assembly
clr.AddReference('rFpro.Controller')

# Import rFpro.controller and some other helpful .NET objects
from rFpro import Controller
from System import DateTime, Decimal

# Create an instance of the rFpro.Controller
#rFpro = Controller(Controller.DefaultPort, '127.255.255.255')
rFpro = Controller.DeserializeFromFile('2kFlatClouds.json')

print(rFpro)

while (rFpro.NodeStatus.NumAlive < rFpro.NodeStatus.NumListeners):
    print(rFpro.NodeStatus.NumAlive, ' of ', rFpro.NodeStatus.NumListeners, ' Listeners connected.')
    time.sleep(1)

print(rFpro.NodeStatus.NumAlive, ' of ', rFpro.NodeStatus.NumListeners, ' Listeners connected.')

rFpro.DynamicWeatherEnabled = True
rFpro.Camera = 'Static001'
rFpro.ParkedTrafficDensity = Decimal(0.5)

for vehicle in ['Accord_Black', 'Accord_Blue', 'Accord_Burgundy', 'Accord_Grey', 'Accord_Silver', 'Accord_White', 'AudiA3_Blue', 'AudiA3_Green', 'AudiA3_Red', 'AudiA3_Silv', 'AudiA3_Sky', 'AudiA3_White', 'AudiA5S_Blue', 'AudiA5S_Grey', 'AudiA5S_Silv', 'AudiA5S_White', 'AudiA5S_Yellow', 'AudiTT_Blue', 'AudiTT_Green', 'AudiTT_Red', 'AudiTT_Silv', 'AudiTT_Sky', 'AudiTT_White', 'ChevroletPickup_Black', 'ChevroletPickup_Blue', 'ChevroletPickup_Grey', 'ChevroletPickup_Red', 'ChevroletPickup_White', 'Coach_Black', 'Coach_Blue', 'Coach_Grey', 'Coach_Red', 'Coach_White', 'CRV_Black', 'CRV_Blue', 'CRV_Burgundy', 'CRV_Grey', 'CRV_White', 'CRV_Yellow', 'DDBus_Black', 'DDBus_Blue', 'DDBus_Grey', 'DDBus_Red', 'DDBus_White', 'Emergency_Ambulance_Germany', 'Emergency_Ambulance_UK', 'Emergency_Ambulance_US', 'Emergency_Police_Germany', 'Emergency_Police_Japan', 'Emergency_Police_UK', 'Emergency_Police_US', 'Fit_Blue', 'Fit_Grey', 'Fit_Red', 'Fit_Silver', 'Fit_White', 'Fit_Yellow', 'Formula_2019', 'Formula_2019_Ghost', 'Formula_2019_H', 'Formula_2019_I', 'Formula_2019_M', 'Formula_2019_S', 'Formula_2019_W', 'Hatchback_AWD_Black', 'Hatchback_AWD_Black_RH', 'Hatchback_AWD_Blue', 'Hatchback_AWD_Blue_RH', 'Hatchback_AWD_Camo', 'Hatchback_AWD_Red', 'Hatchback_AWD_Red_RH', 'Hatchback_AWD_White', 'Hatchback_AWD_White_RH', 'Hatchback_FWD_Black', 'Hatchback_FWD_Blue', 'Hatchback_FWD_Red', 'Hatchback_FWD_White', 'Hatchback_GTI_Black', 'Hatchback_GTI_Blue', 'Hatchback_GTI_Red', 'Hatchback_GTI_White', 'Hatchback_RWD_Black', 'Hatchback_RWD_Blue', 'Hatchback_RWD_Red', 'Hatchback_RWD_White', 'Hatchback-EV_FWD_Red', 'Hatchback-EV_FWD_Red_RH', 'HGV_Kenworth_Blue', 'HGV_Kenworth_Green', 'HGV_Kenworth_White', 'HGV_Kenworth_Yellow', 'HGV_Kenworth+Box01_Red', 'HGV_Kenworth+Box01_Yellow', 'HGV_Kenworth+Box02_Blue', 'HGV_Kenworth+Box02_Green', 'HGV_Kenworth+Box02_Red', 'HGV_Kenworth+Box02_White', 'HGV_Kenworth+Flatbed1_White', 'HGV_Kenworth+Flatbed1_Yellow', 'HGV_Kenworth+Flatbed2_Blue', 'HGV_Kenworth+Flatbed2_Red', 'HGV_Kenworth+Flatbed2_White', 'HGV_Kenworth+Tanker_Black', 'HGV_Kenworth+Tanker_Blue', 'HGV_Kenworth+Tanker_White', 'HGV_Kenworth+Tanker_Yellow', 'HGV_Kenworth+Trailer1_Red', 'HGV_Kenworth+Trailer1_White', 'HGV_Kenworth+Trailer2_Green', 'HGV_Kenworth+Trailer3_Black', 'HGV_Kenworth+Trailer3_Yellow', 'HGV_Kenworth+Transporter_Black', 'HGV_Kenworth+Transporter_Red', 'HGV_Kenworth+Transporter_White', 'HGV_MAN_Black', 'HGV_MAN_Blue', 'HGV_MAN_Green', 'HGV_MAN_Red', 'HGV_MAN_White', 'HGV_MAN_Yellow', 'HGV_MAN+Box01_Blue', 'HGV_MAN+Box01_Green', 'HGV_MAN+Box01_White', 'HGV_MAN+Box01_Yellow', 'HGV_MAN+Box02_Green', 'HGV_MAN+Box02_Red', 'HGV_MAN+Box02_White', 'HGV_MAN+Box02_Yellow', 'HGV_MAN+Flatbed1_Blue', 'HGV_MAN+Flatbed1_White', 'HGV_MAN+Flatbed1_Yellow', 'HGV_MAN+Flatbed2_Black', 'HGV_MAN+Flatbed2_Blue', 'HGV_MAN+Flatbed2_White', 'HGV_MAN+Flatbed2_Yellow', 'HGV_MAN+Tanker_Blue', 'HGV_MAN+Tanker_Chrome', 'HGV_MAN+Tanker_Green', 'HGV_MAN+Tanker_Red', 'HGV_MAN+Tanker_White', 'HGV_MAN+Tanker_Yellow', 'HGV_MAN+Trailer1_Red', 'HGV_MAN+Trailer1_White', 'HGV_MAN+Trailer1_Yellow', 'HGV_MAN+Trailer2_Red', 'HGV_MAN+Trailer2_White', 'HGV_MAN+Trailer3_Blue', 'HGV_MAN+Trailer3_Red', 'HGV_MAN+Trailer3_White', 'HGV_MAN+Trailer3_Yellow', 'HGV_MAN+Transporter_Black', 'HGV_MAN+Transporter_Green', 'HGV_MAN+Transporter_Red', 'HGV_MAN+Transporter_White', 'Hyundai10_White', 'MercedesA_Blue', 'MercedesA_Green', 'MercedesA_Red', 'MercedesA_Silv', 'MercedesA_Sky', 'MercedesA_White', 'MercedesE_Black', 'MercedesE_Blue', 'MercedesE_Brown', 'MercedesE_Grey', 'MercedesE_White', 'MiniBus_Black', 'MiniBus_Blue', 'MiniBus_Grey', 'MiniBus_Red', 'MiniBus_White', 'Peug5008+Caravan_L_White', 'Peug5008+Caravan_S_White', 'Peug5008+CaravanRack_L_White', 'Peug5008+CaravanRack_S_White', 'Peugeot108_Blue', 'Peugeot108_Green', 'Peugeot108_Orange', 'Peugeot108_Red', 'Peugeot108_Sky', 'Peugeot5008_Black', 'Peugeot5008_Blue', 'Peugeot5008_Green', 'Peugeot5008_Grey', 'Peugeot5008_Red', 'PorscheCayman_Blue', 'PorscheCayman_Grey', 'PorscheCayman_Orange', 'PorscheCayman_Red', 'PorscheCayman_Yellow', 'RRS_Blue', 'RRS_Green', 'RRS_Orange', 'RRS_Red', 'RRS_Sky', 'SDBus_Black', 'SDBus_Blue', 'SDBus_Grey', 'SDBus_Red', 'SDBus_White', 'TeslaModelS_Blue', 'TeslaModelS_Cherry', 'TeslaModelS_Green', 'TeslaModelS_Grey', 'TeslaModelS_Sky', 'ToyotaVan_Blue', 'ToyotaVan_Cherry', 'ToyotaVan_White', 'Trailer_Box01', 'Trailer_Box02', 'Trailer_Flatbed1', 'Trailer_Flatbed2', 'Trailer_Hay', 'Trailer_Tanker_Chrome', 'Trailer_Tanker_White', 'Trailer_Trailer1', 'Trailer_Trailer2', 'Trailer_Trailer3', 'Trailer_Transporter', 'Trailer_Water', 'UtilityTractor_Green', 'UtilityTractor+Hay_Blue', 'UtilityTractor+Hay_Green', 'UtilityTractor+Hay_White', 'UtilityTractor+Water_Green', 'UtilityTractorLoader_Green', 'VanDAF_Blue', 'VanDAF_Grey', 'VanDAF_Red', 'VanDAF_White', 'VanDAF_Yellow', 'VanTransit_Black', 'VanTransit_Blue', 'VanTransit_Grey', 'VanTransit_Red', 'VanTransit_White']:
	for location in ['04_HolyheadRoad', '05_CovCentreEast', '07_WarwickCampus', '2kFlat', '2kFlatTrafficLHD', '2kFlatTrafficRHD', 'Bettenfeld', 'ConnecticutRoad', 'Gaydon', 'GermanyK4556', 'HandlingLoop', 'InfiniteHighway', 'LA_Road_Loop', 'Leonberg_Autobahn', 'McityTestFacility', 'MemmingenKempten', 'MillbrookCityHandling', 'MillbrookConstant', 'MillbrookPave', 'Port', 'SimPerfTests', 'Tomei_Expressway', 'VPG_Pro_BlackLake', 'VPG_Pro_BlackLakeXL', 'VPG_Pro_Hills', 'VPG_Pro_LowMU', 'VPG_Pro_Motorway', 'VPG_Pro_RideInputs', 'VPG_Pro_Ring']:
		for dateTime in ['2008-09-03T12:00:00', '2008-09-03T14:00:00']:
			for vehiclePlugin in ['RemoteModelPlugin', 'TrackGeometrySamplePlugin']:
				for cloudLevel in [0, 0.3, 0.6, 1.0]:
					for rainLevel in [0, 0.3, 0.6, 1.0]:
						for fogLevel in [0, 0.1, 0.2]:

							rFpro.Vehicle = vehicle
							rFpro.Location = location
							rFpro.StartTime = DateTime.Parse(dateTime)
							rFpro.VehiclePlugin = vehiclePlugin
							rFpro.Cloudiness = cloudLevel
							rFpro.Rain = rainLevel
							rFpro.Fog = fogLevel
							
							rFpro.StartSession()
							
							# Insert test code here
							time.sleep(5)
							
							rFpro.StopSession()
							

