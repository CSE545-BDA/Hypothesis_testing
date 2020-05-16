import json
from pprint import pprint
rdd = sc.textFile('hdfs:/data/')

complete_data = rdd.map(json.loads).filter(lambda row: 'median_income' in row and 'total_pop' in row and 'geo_id' in row and 'do_date' in row and 'income_per_capita' in row)\
    .map(lambda row: (row['do_date'], (row['geo_id'], row['median_income'], row['total_pop'], row['income_per_capita'])))

per_capita_income_tot_pop = complete_data.map(lambda row:(row[0], (row[1][2] * row[1][3], row[1][2]))) \
    .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])) \
    .map(lambda row: (row[0], row[1][0]/ row[1][1]))

total_population = sc.broadcast(complete_data.filter(lambda row: row[0]=='20062010').map(lambda row: (1, row[1][2])).reduceByKey(lambda a,b:a+b).collect()[0][1])

def get_bottom_40(row):
    lst = sorted(list(row[1]), key=lambda r:r[3])
    running_pop = 0.0
    geo_id_lst = []
    for v in lst:
        if running_pop>0.4*total_population.value:
            break
        running_pop+=v[2]
        geo_id_lst.append(v[0])
    return geo_id_lst

geo_id_lst = complete_data.filter(lambda row: row[0]=='20062010')\
    .groupByKey()\
    .map(get_bottom_40)\
    .collect()

geo_id_set = sc.broadcast(set(geo_id_lst[0]))

per_capita_income_partial_pop = complete_data.filter(lambda row: row[1][0] in geo_id_set.value)\
    .map(lambda row: (row[0], (row[1][2]*row[1][3], row[1][2]))) \
    .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1]))\
    .map(lambda row:(row[0], (row[1][0]/row[1][1])))


per_capita_income_tot_pop_data = per_capita_income_tot_pop.collect()
per_capita_income_partial_pop_data = per_capita_income_partial_pop.collect()


import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt

tot_pop = [('20122016', 29793.506956530848), ('20112015', 28887.455491313358), ('20092013', 28030.88027305989), ('20062010', 27196.386454508654), ('20102014', 28434.410895118297), ('20072011', 27776.975126260993), ('20132017', 31146.495123975765), ('20082012', 27913.392073948147)]

part_pop = [('20122016', 18296.53985100898), ('20112015', 17708.63451800037), ('20092013', 16798.922447681893), ('20062010', 15739.891663421871), ('20102014', 17235.009450435547), ('20072011', 16279.96802027715), ('20132017', 19187.410786742923), ('20082012', 16536.28909251317)]

def clean_data(dt):
    new_dt = []
    for i in range(len(dt)):
        new_dt.append([int(dt[i][0][4:]), float(dt[i][1])])
    return pd.DataFrame(new_dt)

tot_pop = clean_data(tot_pop)
part_pop = clean_data(part_pop)

tot_pop.sort_values(by=[1,0], inplace=True)
part_pop.sort_values(by=[1,0], inplace=True)

# sns.lineplot(data=tot_pop, x=0, y=1, label='Total Population', markers= ["o"])
# sns.lineplot(data=part_pop, x=0, y=1, label='Bottom 40% Population', markers= ["o"])
plt.plot(tot_pop[0], tot_pop[1], linestyle='-', marker='o', label='Total Population')
plt.plot(part_pop[0], part_pop[1], linestyle='-', marker='o', label='Bottom 40% Population')
plt.title('Income Per capita comparison')
plt.xlabel('Year')
plt.ylabel('Income per capita')
plt.legend()
plt.show()

new_list = []
tp = tot_pop.values.tolist()
for i in range(1, len(tot_pop.values.tolist())):
    new_list.append([tp[i][0], (tp[i][1] - tp[i-1][1])/tp[i-1][1]])
    
new_list_half = []
tp = part_pop.values.tolist()
for i in range(1, len(tot_pop.values.tolist())):
    new_list_half.append([tp[i][0], (tp[i][1] - tp[i-1][1])/tp[i-1][1]])
   
top_pop = pd.DataFrame(new_list)
part_pop = pd.DataFrame(new_list_half)

plt.plot(top_pop[0], top_pop[1], linestyle='-', marker='o', label='Total Population')
plt.plot(part_pop[0], part_pop[1], linestyle='-', marker='o', label='Bottom 40% Population')
plt.title('Income Per capita Growth Rate comparison')
plt.xlabel('Year')
plt.ylabel('Income per capita Growth Rate')
plt.legend()


