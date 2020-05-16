import json
import numpy as np
from pprint import pprint

# rdd contains raw text data
rdd = sc.textFile('hdfs:/data/')

# complete data contains rows in (do_date, (geo_id, total_pop, income_per_capita))
unprocessed_data = rdd.map(json.loads).filter(lambda row: 'total_pop' in row \
        and 'geo_id' in row \
        and 'do_date' in row \
        and 'income_per_capita' in row)\
    .map(lambda row: (row['do_date'], \
        (row['geo_id'], row['total_pop'], row['income_per_capita'])))

geoids_to_exclude = sc.broadcast(set(unprocessed_data.map(lambda r: (r[1][0], r[0])).groupByKey().filter(lambda r: len(r[1])!=8).map(lambda r: r[0]).collect()))

complete_data = unprocessed_data.filter(lambda r: r[1][0] not in geoids_to_exclude.value)
all_years = sc.broadcast(sorted(complete_data.map(lambda r:r[0]).distinct().collect()))

# contains total population 
total_population = sc.broadcast(dict(complete_data.map(lambda row: (row[0], row[1][1]))\
    .reduceByKey(lambda a,b:a+b)\
    .collect()))

# cotains minimum income of the population
min_income = sc.broadcast(dict(complete_data.map(lambda r: (r[0], r[1][2]))\
    .reduceByKey(lambda a,b: min(a,b))\
    .collect()))

# contains maximum income of the population
max_income = sc.broadcast(dict(complete_data.map(lambda r: (r[0], r[1][2]))\
    .reduceByKey(lambda a,b: max(a,b))\
    .collect()))

def checkIfUnfinished():
    for year in min_income.value:
        if min_income.value[year]<=max_income.value[year]:
            return True
    return False

def retrieveMid():
    mid_income = dict()
    for year in min_income.value:
        if min_income.value[year]<=max_income.value[year]:
            mid_income[year] = int(max_income.value[year]*0.4 + min_income.value[year]*0.6)
    return mid_income

# strictly upper
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

# income_per_capita_upper_margin = {'20122016': 23462, '20112015': 22763, '20092013': 22222, '20062010': 21617, '20102014': 22511, '20072011': 22035, '20132017': 24531, '20082012': 22157}
income_per_capita_upper_margin = sc.broadcast(income_per_capita_upper_margin)

# first hypothesis

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
# 0.23147445379849582


# second hypothesis

first_year_poor_pop = sc.broadcast(set(complete_data.filter(lambda r: r[0]=='20062010' and r[1][2]>=income_per_capita_upper_margin.value['20062010']).map(lambda row:row[1][0]).collect()))

def growth_rate_calc(row):
    info = sorted(list(row[1]))
    retList = []
    for i in range(len(all_years.value)-1):
        retList.append((("gr_"+str(all_years.value[i]), row[0]),((info[i+1][1] - info[i][1])/info[i][1])))
    return retList

def second_hypothesis_bloat(row):
    retList = []
    for i in range(len(all_years.value)-1):
        if row[0][0] == "gr_"+str(all_years.value[i]):
            if i>0 and i<len(all_years.value)-2:
                retList.append(((i, row[0][1]), (1,row[1])))
                retList.append(((i+1, row[0][1]), (0,row[1])))
            if i==0:
                retList.append(((i+1, row[0][1]), (0, row[1])))
            if i==len(all_years.value)-2:
                retList.append(((i, row[0][1]), (1,row[1])))
            break
    return retList

def create_diff(r):
    row = list(r[1])
    diff = 0
    for entry in row:
        if entry[0] == 0: diff += entry[1]
        else: diff -= entry[1]
    return (r[0][0], (diff, diff**2, 1))

def calc_tvalues(row):
    sum_val = row[1][0]
    sum_sqval = row[1][1]
    cnt = row[1][2]
    moment1 = sum_val/cnt
    moment2 = sum_sqval/cnt
    sd = (moment2 - (moment1**2))**0.5
    tval = moment1/(sd/(cnt**0.5))
    return (row[0], tval)

# y-o-y growth rate (('gr_STARTYEAR', 'GEOID'), GROWTH_RATE)
second_hypothesis_domain = complete_data.filter(lambda r: r[1][0] in first_year_poor_pop.value) \
    .map(lambda r: (r[1][0], (r[0], r[1][2]))) \
    .groupByKey() \
    .flatMap(growth_rate_calc) \
    .flatMap(second_hypothesis_bloat) \
    .groupByKey() \
    .map(create_diff)


second_hypothesis_results = second_hypothesis_domain.reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1], a[2]+b[2])) \
    .map(calc_tvalues)\
    .collect()
# [(1, 27.806014256188217), (2, 2.3644756942115355), (3, -20.295666635081272), (4, 3.1254629114528214), (5, -49.86682299225638), (6, -26.117475147504038)]

dof = len(first_year_poor_pop.value) - 1
tvalue_005 = 1.6449


# third hypothesis

def calc_tvalues_th_hyp(row):
    sum_val = row[1][0]
    sum_sqval = row[1][1]
    cnt = row[1][2]
    moment1 = sum_val/cnt 
    moment2 = sum_sqval/cnt 
    sd = (moment2 - (moment1**2))**0.5
    tval = (moment1 - h_not.value['gr_'+row[0]])/(sd/(cnt**0.5))
    return (row[0], tval)

# (do_date, (geo_id, total_pop, income_per_capita))
pop_gr_rt = complete_data.map(lambda r: (r[1][0], (r[0], r[1][2]))) \
    .groupByKey() \
    .flatMap(growth_rate_calc)

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
    .map(calc_tvalues_th_hyp)

t_values.collect()
# [('20122016', 31.377913250595373), ('20062010', 23.297626118248537), ('20102014', 29.867328318185756), ('20082012', 24.49310238717426), ('20112015', 31.57440145632035), ('20092013', 28.745102726264847), ('20072011', 28.645126095540515)]

dof = third_hyp_distribution.countByKey()
# {'20062010': 89952, '20072011': 90194, '20082012': 90143, '20092013': 90107, '20102014': 90228, '20112015': 90380, '20122016': 90604})


##########################################
dict(complete_data.filter(lambda r: r[1][2]<=income_per_capita_upper_margin.value[r[0]]) \
    .map(lambda r: (r[0], r[1][0])) \
    .groupByKey() \
    .mapValues(len).collect())


poor_pop_gr_rt = complete_data.filter(lambda r: r[1][2]<=income_per_capita_upper_margin.value[r[0]])\
    .map(lambda r: (r[1][0], (r[0], r[1][2]))) \
    .groupByKey() \
    .flatMap(growth_rate_calc)

poor_pop_gr_rt = complete_data.filter(lambda r: r[1][2]<=income_per_capita_upper_margin.value[r[0]])\
    .map(lambda r: (r[1][0], (r[0], r[1][2])))\
    .groupByKey()\
    .flatMap(growth_rate_calc)   
###

final_rdd = complete_data.filter(lambda row: row[1][2]<=income_per_capita_upper_margin.value[row[0]]) \
    .map(lambda row: (row[0], (row[1][1], row[1][1]*row[1][2]))) \
    .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])) \
    .map(lambda row: (row[0], row[1][1]/row[1][0]))

# final_rdd.collect()
# [('20122016', 16873.90751180781), ('20112015', 16346.06572129993), ('20092013', 15970.429434225625), ('20062010', 15628.308802509884), ('20102014', 16166.947058261345), ('20072011', 15923.119587059546), ('20132017', 17666.22286617404), ('20082012', 15961.622197883431)]



# complete_data.map(lambda r: (r[1][0], r[0])).groupByKey().mapValues(len).filter(lambda r: r[1]!=8).count()
# 556 - geoids with data not available in all years

# complete_data.map(lambda r: (r[1][0], r[0])).groupByKey().count()
# 219358 - total geoids










