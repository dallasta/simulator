import numpy as np

'''
Settings for town generation
'''

town_name = 'Turin' 

# Make sure to download country-specific population density data
# from https://data.humdata.org/organization/facebook
population_path='lib/data/population/population_ita_2019-07-01.csv' # Population density file

sites_path='lib/data/queries/' # Directory containing OSM site query details
bbox = (44.955809, 45.153232, 7.434998, 7.889557) # Coordinate bounding box

# Population per age group in the region (matching the RKI age groups)
# Source for Germany: https://www.citypopulation.de/en/germany/
population_per_age_group = np.array([
   32714,  # 0-4
    72941,  # 5-14
    170080,  # 15-34
    319600,  # 35-59
    201533,  # 60-79
    78830]) # 80+

town_population = 875698 
region_population = population_per_age_group.sum()

# Daily testing capacity: approx. average over last 18 days 01/05/20-18/05/20 (from https://covid19.lbreda.com/region/1 )
daily_tests_unscaled = 1000 #roughly 5000 per day in the region

# Information about household structure (set to None if not available)
# Source for Europe: https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Household_composition_statistics&oldid=307948
household_info = {
    'size_dist' : [33.2, 27.1, 19.3, 15.1, 5.3], # distribution of household sizes (1-5 people)
    'soc_role' : {
        'children' : [1, 1, 3/20, 0, 0, 0], # age groups 0,1,2 can be children 
        'parents' : [0, 0, 17/20, 1, 0, 0], # age groups 2,3 can be parents
        'elderly' : [0, 0, 0, 0, 1, 1] # age groups 4,5 are elderly
    }
}

