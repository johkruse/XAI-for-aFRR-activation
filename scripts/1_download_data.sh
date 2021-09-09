cd ../../External_data/ENTSO-E

# To access the sftp you need an account for the ENTSO-E Transparency platform 
sftp jo.kruse@fz-juelich.de@sftp-transparency.entsoe.eu << EOF
cd TP_export

# Actual load 
cd ActualTotalLoad_6.1.A
get 2019*.csv
get 202[0-1]*.csv

# Load forecast
cd ..
cd DayAheadTotalLoadForecast_6.1.B
get 2019*.csv
get 202[0-1]*.csv

# Generation forecast
cd ..
cd DayAheadAggregatedGeneration_14.1.C
get 2019*.csv
get 202[0-1]*.csv

# Generation per type
cd ..
cd AggregatedGenerationPerType_16.1.B_C
get 2019*.csv
get 202[0-1]*.csv

# Wind/solar forecast
cd ..
cd DayAheadGenerationForecastForWindAndSolar_14.1.D
get 2019*.csv
get 202[0-1]*.csv

# Day ahead prices
cd ..
cd DayAheadPrices_12.1.D
get 2019*.csv
get 202[0-1]*.csv

# Day-ahead scheduled exchanges
cd ..
cd DayAheadCommercialSchedules_12.1.F
get 2019*.csv
get 202[0-1]*.csv

# Total scheduled exchanges
cd ..
cd TotalCommercialSchedules_12.1.F
get 2019*.csv
get 202[0-1]*.csv

# Physical cross-border flow
cd ..
cd PhysicalFlows_12.1.G
get 2019*.csv
get 202[0-1]*.csv

EOF 
