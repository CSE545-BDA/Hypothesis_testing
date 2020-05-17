"""
    Python file for hypothesis testing

    Underlying idea to test: Growth rate of bottom 40 percentile population 
    based on income per capita V/S growth rate of entire population

    Team: UBI (Big Data Analytics - CSE545) - Stony Brook University
"""

from pyspark import SparkContext
import json
import numpy as np
from pprint import pprint
import time

global sc, complete_data, income_per_capita_upper_margin, all_years, total_population

def find40tile():
    '''
        Novel approach for finding nth (here, n = 40) percentile of a dataset in distributed environment (mapreduce/spark)
        Modified binary search algorithm to find the next largest element
        
        Steps:
        1. Finds minimum and maximum income per capita for each year
        2. assumes 40th percentile of population to earn <= (0.4 * max_income_per_capita + 0.6 * min_income_per_capita)
        3. finds count of population below the assummed value of income 
        4. if population count is less that 40% of total population, assume new min to be the assumed value + 1, else
            assume new max = assumed value - 1
        5. Go to step 2 until min>max
        6. correct output is the last assumed value (40 percentile)

        This algorithm avoids sorting the entire data. In our case, it was able to handle this. However, the logic 
            wouldn't be fullproof when the size of data increases in the future. And this, speed is traded off for scalability.
    '''
    # cotains minimum income of the population
    min_income = sc.broadcast(dict(complete_data.map(lambda r: (r[0], r[1][2]))\
        .reduceByKey(lambda a,b: min(a,b))\
        .collect()))

    # contains maximum income of the population
    max_income = sc.broadcast(dict(complete_data.map(lambda r: (r[0], r[1][2]))\
        .reduceByKey(lambda a,b: max(a,b))\
        .collect()))

    def checkIfUnfinished():
        '''
            Checks if the binary search iteration is completed.
            Returns True even if a single year has min income_per_capita <= max income_per_capita
        '''
        for year in min_income.value:
            if min_income.value[year]<=max_income.value[year]:
                return True
        return False

    def retrieveMid():
        '''
            Returns the mid value (0.4 * max_income_per_capita + 0.6 * min_income_per_capita) for each year
        '''
        mid_income = dict()
        for year in min_income.value:
            if min_income.value[year]<=max_income.value[year]:
                mid_income[year] = int(max_income.value[year]*0.4 + min_income.value[year]*0.6)
        return mid_income

    income_per_capita_upper_margin = retrieveMid()
    i=1
    while checkIfUnfinished():
        mid_income = retrieveMid()
        pprint("ITERATION: " + str(i))
        print("MID_INCOME: ", mid_income)
        print("income_per_capita_upper_margin: ", income_per_capita_upper_margin)
        print("MIN_INCOME: ", min_income.value)
        print("MAX_INCOME: ", max_income.value)
        i+=1
        filter_range = set(min_income.value) - set(mid_income)
        lower_population = dict(complete_data.filter(lambda row: row[0] not in filter_range and row[1][2]<=mid_income[row[0]]) \
            .map(lambda row: (row[0], row[1][1])) \
            .reduceByKey(lambda a,b: a+b)\
            .collect())
        print("LOWER_POPULATION: ", lower_population)
        temp_min_inc = min_income.value.copy()
        temp_max_inc = max_income.value.copy()
        for year in lower_population:
            if lower_population[year]<=int(0.4 * total_population.value[year]):
                temp_min_inc[year] = mid_income[year]+1
                temp_max_inc[year] = max_income.value[year]
            else:
                income_per_capita_upper_margin[year] = mid_income[year]
                temp_max_inc[year] = mid_income[year]-1
                temp_min_inc[year] = min_income.value[year]
        min_income = sc.broadcast(temp_min_inc)
        max_income = sc.broadcast(temp_max_inc)
        print("===============================================================")
    print("income_per_capita_upper_margin: ", income_per_capita_upper_margin)
    # income_per_capita_upper_margin = {'20122016': 23462, '20112015': 22763, '20092013': 22222, '20062010': 21617, '20102014': 22511, '20072011': 22035, '20132017': 24531, '20082012': 22157}
    return income_per_capita_upper_margin

def growth_rate_calc(row, ay):
    '''
        Input format: row = (geoid, [(year_info, income_per_capita)...]) - contains all years' information
                      ay = orderedlist of all available years
        Output format: [(("gr_"+first_year, geoid), growth_rate_from_firstyr_to_next_yr )...]

        Formula for growth rate = (new_income_per_capita - old_income_per_capita)/old_income_per_capita
    '''
    info = sorted(list(row[1]))
    retList = []
    for i in range(len(ay)-1):
        retList.append((("gr_"+str(ay[i]), row[0]),((info[i+1][1] - info[i][1])/info[i][1])))
    return retList

def second_hypothesis_bloat(row, ay):
    '''
        For our second analysis, we have multiple hypotheses running in parallel. This function 
            prepares data for individual hypothesis, by tagging each data point by individual
            hypothesis' identifier. 
        Input: row = (("gr_"+first_year, geoid), growth_rate_from_firstyr_to_next_yr )
               ay = orderedlist of all available years
        Output: [((hypothesis_id, geoid),(sample_id, growth_rate_from_firstyr_to_next_yr)) ... ]
    '''
    retList = []
    for i in range(len(ay)-1):
        if row[0][0] == "gr_"+str(ay[i]):
            if i>0 and i<len(ay)-2:
                retList.append(((i, row[0][1]), (1,row[1])))
                retList.append(((i+1, row[0][1]), (0,row[1])))
            if i==0:
                retList.append(((i+1, row[0][1]), (0, row[1])))
            if i==len(ay)-2:
                retList.append(((i, row[0][1]), (1,row[1])))
            break
    return retList

def create_diff(r):
    '''
        Creates the difference distribution for running t-test
        Input: ((hypothesis_id, geoid), [(0/1, growth_rate_from_firstyr_to_next_yr), 
            (1/0, growth_rate_from_firstyr_to_next_yr)])
        Output: (hypothesis_id, (difference, difference**2, 1))
    '''
    row = list(r[1])
    diff = 0
    for entry in row:
        if entry[0] == 0: diff += entry[1]
        else: diff -= entry[1]
    return (r[0][0], (diff, diff**2, 1))

def calc_tvalues(row):
    '''
        Calculates t_values for individual hypothesis in second analysis (paired ttest)
            Analysis is done on difference distribution
        Input: (hypothesis_id, (sum_of_values, sum_of_sq_values, sample_count))
        Output: (hypothesis_id, t_value)
        Formula: tvalue = mean/(standard_deviation/(sample_count**0.5))
    '''
    sum_val = row[1][0]
    sum_sqval = row[1][1]
    cnt = row[1][2]
    moment1 = sum_val/cnt
    moment2 = sum_sqval/cnt
    sd = (moment2 - (moment1**2))**0.5
    tval = moment1/(sd/(cnt**0.5))
    return (row[0], tval)

def calc_tvalues_th_hyp(row, h_not):
    '''
        Calculates t_values for individual hypothesis in third analysis (one-tailed ttest)
        Input: row: (hypothesis_id, (sum_of_values, sum_of_sq_values, sample_count))
               h_not: {hypothesis_id: proposed_hypothesis_value, ...}
        Output: (hypothesis, t_value)
        Formula: tvalue = (mean - proposed_hypothesis_value)/(standard_deviation/(sample_count**0.5))
    '''
    sum_val = row[1][0]
    sum_sqval = row[1][1]
    cnt = row[1][2]
    moment1 = sum_val/cnt 
    moment2 = sum_sqval/cnt 
    sd = (moment2 - (moment1**2))**0.5
    tval = (moment1 - h_not.value['gr_'+row[0]])/(sd/(cnt**0.5))
    return (row[0], tval)

def first_hypothesis():
    '''
        First analysis: 
        1. Find mean income per capita of both bottom 40 percentile and total population
            mean_income_per_capita = sum_over_all_geoids(population * income_per_capita)/sum_over_all_geoids(population)
                for each year
        2. Find year-over-year growth for both bottom 40 percentile and total population
            year-over-year_growth = (nextyr_mean_income_per_capita - currentyr_mean_income_per_capita)/currentyr_mean_income_per_capita
        3. The above steps gives us 2 dependent distributions viz bottom 40 percentile 
            and total population growth rate
        4. Run paired ttest with the following hypothesis
            H_0: Growth rate of 40 tile < Growth rate of total population
            H_1: Growth rate of 40 tile >= Growth rate of total population
            4.1. Create difference distribution with (old_growth_rate - new_growth_rate) 
            4.2. Find tvalue = mean/((variance/sample_count)**0.5)
        5. If we can reject the null hypothesis, then it would be an indicator of reduced inequalities over the years
        6. Output: tvalue, degree of freedom
    '''
    poor_pop = complete_data.filter(lambda row: row[1][2]<=income_per_capita_upper_margin.value[row[0]]) \
        .map(lambda row: (row[0], (row[1][1], row[1][1]*row[1][2]))) \
        .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])) \
        .map(lambda row: (row[0], row[1][1]/row[1][0])) \
        .collect()
    poor_pop = sc.broadcast(poor_pop)
    # [('20122016', 16882.095751975965), ('20112015', 16355.702860873493), ('20092013', 15978.678258285534), ('20062010', 15635.599621689878), ('20102014', 16175.68845392017), ('20072011', 15931.285858687235), ('20132017', 17674.51860188616), ('20082012', 15970.1425140269)]

    total_pop = complete_data.map(lambda row: (row[0], (row[1][1], row[1][1]*row[1][2]))) \
        .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])) \
        .map(lambda row: (row[0], row[1][1]/row[1][0])) \
        .collect()
    total_pop = sc.broadcast(total_pop)
    # [('20122016', 29635.96686669512), ('20112015', 28739.56719951025), ('20092013', 27961.014209276225), ('20062010', 27132.668149443776), ('20102014', 28363.36571720921), ('20072011', 27713.47284846637), ('20132017', 30979.16123129314), ('20082012', 27852.416570353675)]

    poor_pop_growth_rate = []
    total_pop_growth_rate = []
    for i in range(len(poor_pop.value)-1):
        poor_pop_growth_rate.append((poor_pop.value[i+1][1] - poor_pop.value[i][1])/poor_pop.value[i][1])
        total_pop_growth_rate.append((total_pop.value[i+1][1] - total_pop.value[i][1])/total_pop.value[i][1])

    poor_pop_growth_rate = np.array(poor_pop_growth_rate)
    total_pop_growth_rate = np.array(total_pop_growth_rate)
    diff_growth_rate = poor_pop_growth_rate - total_pop_growth_rate
    t_value = np.average(diff_growth_rate)/((np.var(diff_growth_rate)/diff_growth_rate.shape[0])**0.5)
    return (t_value, diff_growth_rate.shape[0]-1)

def second_hypothesis():
    '''
        Second analysis:
        Analysis only on bottom 40 tile population to investigate if their growth rate of income per capita
            hass been growing year-over-year. 
            <emphasis>ANALYSIS TO CHECK IF THE GROWTH RATE HAS INCREASED, AND NOT INCOME_PER_CAPITA</emphasis>
        For the analysis, we have only considered geoids which were in the bottom 40 percentile during the first
            year of data (20062010).
        1. Find geoids on the bottom 40 tile in 20062010.
        2. For the geoids, calculate their growth rate y-o-y (i.e. 20062010 to 20072011, 20072011 to 20082012)
        3. Run paired ttest for a pair of adjacent y-o-y growth rates (gr_20062010 vs gr_20072011)
            Here we have multiple hypothesis to be executed in parallel. List of hypothesis in current data
            a. gr_20062010 vs gr_20072011 (income per capita growth rate from 20062010 to 20072011 
                                            vs income per capita growth rate from 20072011 to 20082012)
            b. gr_20072011 vs gr_20082012
            c. gr_20082012 vs gr_20092013
            d. gr_20092013 vs gr_20102014
            e. gr_20102014 vs gr_20112015
            f. gr_20112015 vs gr_20122016
            g. gr_20122016 vs gr_20132017
        4. H_0: Average growth rate of current year <= average growth rate of next year
           H_1: Average growth rate of current year > average growth rate of next year
        5. If the null hypothesis is rejected consistently, then it would be an indicator of upliftment of 
            poorer population
        6. Output: (hypothesis_output, degree of freedom)
    '''
    first_year_poor_pop = sc.broadcast(set(\
        complete_data.filter(lambda r: r[0]=='20062010' and r[1][2]<=income_per_capita_upper_margin.value['20062010'])\
            .map(lambda row:row[1][0])\
            .collect()))

    ay = all_years.value.copy()
    first_yr_ = first_year_poor_pop.value

    # y-o-y growth rate (('gr_STARTYEAR', 'GEOID'), GROWTH_RATE)
    second_hypothesis_domain = complete_data.filter(lambda r: r[1][0] in first_yr_) \
        .map(lambda r: (r[1][0], (r[0], r[1][2]))) \
        .groupByKey() \
        .flatMap(lambda r: growth_rate_calc(r, ay)) \
        .flatMap(lambda r: second_hypothesis_bloat(r, ay)) \
        .groupByKey() \
        .map(lambda r: create_diff(r))

    second_hypothesis_results = second_hypothesis_domain.reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1], a[2]+b[2])) \
        .map(lambda r: calc_tvalues(r))\
        .collect()
    # [(1, 27.806014256188217), (2, 2.3644756942115355), (3, -20.295666635081272), (4, 3.1254629114528214), (5, -49.86682299225638), (6, -26.117475147504038)]

    dof = len(first_yr_) - 1
    tvalue_005 = 1.6449
    return second_hypothesis_results, dof

def third_hypothesis():
    '''
        Third analysis:
        For each yearly growth rate of income per capita, we compare if the bottom 40tile has had 
            more growth rate as compared to total population
        1. For each yearly growth rate of total population, we find their mean(MLE). This would be our 
            hypothesized proposed values (h_not, hereon) for the bottom 40tile analysis.
        2. We find bottom 40 tile population (geoids) based on their income per capita. Then find 
            their growth rates for the particular year.
        3. We run one-tailed ttest on the bottom 40 tile population with the following hypothesis:
            H_0: Growth rate of bottom 40 tile<=Growth rate of total population (h_not)
            H_1: Growth rate of bottom 40 tile>Growth rate of total population (h_not)
        4. This requires multiple hypothesis to be executed in parallel. For our current data, the 
            exhaustive list of hypotheses:
            a. Growth rate of 20062010 <= H_not(20062010) - H_not refers to mean of total population
                                                            for that specific year
            a. Growth rate of 20072011 <= H_not(20072011) 
            a. Growth rate of 20082012 <= H_not(20082012) 
            a. Growth rate of 20092013 <= H_not(20092013) 
            a. Growth rate of 20102014 <= H_not(20102014) 
            a. Growth rate of 20112015 <= H_not(20112015) 
            a. Growth rate of 20122016 <= H_not(20122016) 
        5. If the null hypothesis is consistenly rejected, it would mean that growth rate of bottom 40tile
            population was consistently more than total population, indicating reduced inequality.
        6. Output: (t_values, degree_of_freedom)
    '''
    # (do_date, (geo_id, total_pop, income_per_capita))
    pop_gr_rt = complete_data.map(lambda r: (r[1][0], (r[0], r[1][2]))) \
        .groupByKey() \
        .flatMap(lambda r: growth_rate_calc(r, all_years.value))

    h_not = sc.broadcast(dict(pop_gr_rt.map(lambda r: (r[0][0], (r[1], 1)))\
        .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1]))\
        .map(lambda r: (r[0], r[1][0]/r[1][1]))\
        .collect()))
    # {'gr_20072011': 0.012294250261111007, 'gr_20112015': 0.03918944389998251, 'gr_20102014': 0.02042497083252045, 'gr_20062010': 0.02833325764075963, 'gr_20092013': 0.021714268571786367, 'gr_20082012': 0.011551676204223869, 'gr_20122016': 0.052591302577792015}
    third_hyp_distribution = complete_data.filter(lambda r: r[1][2]<=income_per_capita_upper_margin.value[r[0]]) \
        .map(lambda r: ((r[0], r[1][0]), r[1][2])) \
        .join(complete_data.map(lambda r: ((str(int(r[0][:4])-1)+str(int(r[0][4:])-1), r[1][0]),r[1][2]))) \
        .map(lambda r: (r[0][0], (r[1][1] - r[1][0])/r[1][0]))

    t_values = third_hyp_distribution.map(lambda r: (r[0], (r[1], r[1]**2, 1)))\
        .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1], a[2]+b[2]))\
        .map(lambda r: calc_tvalues_th_hyp(r, h_not))

    t_val = t_values.collect()
    # [('20122016', 31.377913250595373), ('20062010', 23.297626118248537), ('20102014', 29.867328318185756), ('20082012', 24.49310238717426), ('20112015', 31.57440145632035), ('20092013', 28.745102726264847), ('20072011', 28.645126095540515)]

    dof = third_hyp_distribution.countByKey()

    return t_val, dof

if __name__ == '__main__':
    t1 = time.time()
    sc = SparkContext("yarn", "HypothesisTesting")
    rdd = sc.textFile('hdfs:/data/')
    unprocessed_data = rdd.map(json.loads).filter(lambda row: 'total_pop' in row \
            and 'geo_id' in row \
            and 'do_date' in row \
            and 'income_per_capita' in row)\
        .map(lambda row: (row['do_date'], \
            (row['geo_id'], row['total_pop'], row['income_per_capita'])))

    # Data cleaning - removing geoids with incomplete information
    geoids_to_exclude = sc.broadcast(set(unprocessed_data.map(lambda r: (r[1][0], r[0]))\
        .groupByKey()\
        .filter(lambda r: len(r[1])!=8)\
        .map(lambda r: r[0])\
        .collect()))

    # complete data contains rows in (do_date, (geo_id, total_pop, income_per_capita))
    complete_data = unprocessed_data.filter(lambda r: r[1][0] not in geoids_to_exclude.value)
    
    # ordered list of all available years of information
    all_years = sc.broadcast(sorted(complete_data.map(lambda r:r[0]).distinct().collect()))

    # contains total population 
    total_population = sc.broadcast(dict(complete_data.map(lambda row: (row[0], row[1][1]))\
        .reduceByKey(lambda a,b:a+b)\
        .collect()))

    income_per_capita_upper_margin = find40tile()
    # O/P: {'20122016': 23462, '20112015': 22763, '20092013': 22222, '20062010': 21617, '20102014': 22511, '20072011': 22035, '20132017': 24531, '20082012': 22157}
    income_per_capita_upper_margin = sc.broadcast(income_per_capita_upper_margin)
    
    t2 = time.time()
    print("INIT Time: ", t2-t1)
    print("First_Hypothesis: ", first_hypothesis())
    t3 = time.time()
    print("First hypothesis time: ", t3-t2)
    print("Second Hypothesis: ", second_hypothesis())
    t4 = time.time()
    print("Second hypothesis time: ", t4-t3)
    print("third_hypothesis: ", third_hypothesis())
    t5 = time.time()
    print("Third hypothesis time: ", t5-t4)
