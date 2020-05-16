import json
from pprint import pprint
rdd = sc.textFile('hdfs:/data/')

complete_data = rdd.map(json.loads).filter(lambda row: 'median_income' in row and 'total_pop' in row and 'geo_id' in row and 'do_date' in row and 'income_per_capita' in row)\
    .map(lambda row: (row['do_date'], (row['median_income'], row['total_pop'])))

median_income_tot_pop = complete_data.map(lambda row:(row[0], (row[1][0] * row[1][1], row[1][1]))) \
    .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])) \
    .map(lambda row: (row[0], row[1][0]/ row[1][1]))

total_population = sc.broadcast(dict(complete_data.reduceByKey(lambda a,b: (0, a[1]+b[1])).map(lambda row: (row[0], row[1][1]*0.4)).collect()))

def get_40_poor(row):
    running_pop = 0.0
    running_income = 0.0
    for v in sorted(row[1], key=lambda k:k[1]):
        if running_pop>total_population.value[row[0]]:
            break
        running_pop += v[1]
        running_income += (v[0]*v[1])
    return (row[0], running_income/running_pop)

below_40_calc = complete_data.groupByKey().map(get_40_poor).collect()
total_pop_calc = median_income_tot_pop.collect()

pprint(sorted(below_40_calc, key=lambda r:r[0]))
pprint(sorted(total_pop_calc, key=lambda r:r[0]))
