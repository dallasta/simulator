import numpy as np

'''
Settings for town generation
'''

town_name = 'Isle_of_Wight' 

# Make sure to download country-specific population density data
# from https://data.humdata.org/organization/facebook
population_path='lib/data/population/population_gbr_2019-07-01.csv' # Population density file

sites_path='lib/data/queries/' # Directory containing OSM site query details
bbox = (50.5741,50.7695,-1.5532,-1.0616) # Coordinate bounding box

# Population per age group in the region (matching the RKI age groups)
# Source for UK: https://www.ons.gov.uk/
population_per_age_group = np.array([
    6220,  # 0-4
    14380,  # 5-14
    26699,  # 15-34
    43852,  # 35-59
    40047,  # 60-79
    10573]) # 80+

town_population = 141771 
region_population = population_per_age_group.sum()

# Daily testing capacity per 100k people
# Roughly 100k tests per day in Germany: https://www.rki.de/DE/Content/Infekt/EpidBull/Archiv/2020/Ausgaben/15_20.pdf?__blob=publicationFile
daily_tests_unscaled = 300

# Information about household structure (set to None if not available)
# Source for UK: https://www.ons.gov.uk/ 
household_info = {
    'size_dist' : [29.5, 34.5, 15.4, 14.0, 6.6], # distribution of household sizes (1-5 people)
    'soc_role' : {
        'children' : [1, 1, 3/20, 0, 0, 0], # age groups 0,1,2 can be children 
        'parents' : [0, 0, 17/20, 1, 0, 0], # age groups 2,3 can be parents
        'elderly' : [0, 0, 0, 0, 1, 1] # age groups 4,5 are elderly
    }
}

