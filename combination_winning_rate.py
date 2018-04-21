from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import size
from pyspark import SparkContext, SparkConf
import math

conf = SparkConf().set("local", "false")
sc = SparkContext(appName="PythonStatusAPIDemo", conf=conf)

rdd = sc.textFile('games.csv')


def win_1(row):
    r = math.pow(int(row[4])-2,2)
    return r


def win_2(row):
    r = int(row[4])-1
    return r


d_split = rdd.map(lambda x: x.split(','))
d_split = d_split.filter(lambda x: x[0] != 'gameId')
d_frame = d_split.map(lambda x: Row(id=x[0], items=x[11:30][:15:3], winner=win_1(x)))
d_frame_2 = d_split.map(lambda x: Row(id=x[0], items=x[36:50][:15:3], winner=win_2(x)))

champion = ['497', '498', '18', '412', '67', '40', '141']
combination = [['497', '498'],
               ['18', '412'],
               ['412', '67'],
               ['40', '18'],
               ['141', '18']]

spark = SparkSession.builder.appName('data').getOrCreate()
df_1 = spark.createDataFrame(d_frame)
df_2 = spark.createDataFrame(d_frame_2)

df = df_1.union(df_2).collect()

for i in champion:
    term = df
    term = list(filter(lambda x: i in x['items'], term))
    bottom = len(term)
    up = len(list(filter(lambda x: x['winner'] is 1, term)))
    print(up)
    result = up/bottom
    print('winning rate of %s is %f' % (i, result))

for i in combination:
    term = df
    term = list(filter(lambda x: i[0] in x['items'] and i[1] in x['items'], term))
    bottom = len(term)
    up = len(list(filter(lambda x: x['winner'] == 1.0, term)))
    result = up/bottom
    print('winning rate of %s is %f' % (i[0]+' '+i[1], result))

