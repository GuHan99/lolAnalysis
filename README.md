# lolAnalysis

## Abstract

This project aims at finding useful tactics for game of League of Legends. Basic idea is to get useful champion combinations to increase winning rate. Besides, the analysis will include other related winning factors. So the players of this game will get some useful tips extrapolated from stochastic analysis.

## I. Introduction

### 1. League of legends
League of legends is an online MOBA game. Ten players divided into 5V5 choose ten distinct champions as themselves. Each player’s goal is to grab more resources, take more advantage for his team. When resources and advantage accumulated to a certain level against the other side, the enemy team’s nexus will crash. So they win.

Some related terms which would be used in this report:

Tower, inhibitor:
These two are buildings guarding outsides of nexus. They are important resources for winning.

Rift herald, dragon, baron:
These are monsters within the map. Killing them will be entitled strengths for winning. Also useful resources.

Marksman and support:
Two team character types. Each team is composed of 5 fixed positions, top lane, mid lane, jungle, marksman and support. Marksman and support are working together most of the time so their relationship is the closest.

### 2. Problem to solve
First, find frequent champion combinations from match history. 
Second, find out if these combinations concluded from problem1 would increase the winning rate, these combinations are useful or not.

## II. Materials and Methods

### 1. Dataset
The dataset which would be used in this analysis is a collection of 51490 matches. Each row stands for a match record. Each record is made up of 61 columns, gameId, winner, firstBlood etc.

This is a brief preview of the dataset csv file.




### 2.Technologies and algorithms related

a. Frequent itemsets
The method is used for gathering frequent champion combinations. Because this analysis is not aimed at finding possible teammates’ champion choices and their corresponding possibilities, finding frequent combinations is sufficient for the next step analysis. So associations rules will not be considered in this report. Choosing support is important in this analysis.
In pyspark ml libraries, frequent itemsets are conducted by FP-Growth model. Set up minSupport and itemCol and the model will produce the result.

b. Naïve Bayes classifier
This method is to predict the winning team. Suppose the selected factors are conditionally independent, naïve bayes classifier would multiply each factors and choose the highest score as the winning team. 

## III. Results

### 1. Frequent teammate combinations
To take full use of the data, each match has two teams. So each row would be divided into two baskets. Union method of dataframe would be used here to put team2’s basket below team1’s.

League of Legends rules that, each player can invite at most one player as teammate. So the result frequent itemset size of 2 would be considered valid. Because a third teammate is less likely to choose helpful champions.

Minsupport choosing is relevant to each champion’s attendance. League of Legends have 138 champions (until this dataset’s creating). So basically each champion has an average attendance of  . Minsupport should be accordingly larger than that. 
According to other League of Legends analysis, popular champions’ attendance is around 10%. Two-champions combination’s attendance should be less than that because even both popular champions, the attendance is 0.1*0.1 = 1%. Less than 1% would be much better. 
So minsupport chosen as 0.001.

Items[]
freq
Items[]
freq
497, 498
2362
64, 18
1112
18, 412
1541
29, 412
1018
67, 412
1372
40, 67
1014
40, 18
1266
141, 67
1013
141, 18
1207
117, 18
1011
141, 412
1176
222, 412
1009
64, 412
1145


202, 412
1139


236, 412
1132


Each items contains two champion Ids map to a champion.
Take first few rows to validate.
497-> “The Charmer Rakan”
498-> “The Rebel Xayah”
It makes sense because this is a marksman+support combination. And most importantly, these two champion characters are couple birds according to the LOL background story.

64-> “the Blind Monk Lee Sin”
412-> “The Chain Warden Thresh”
It also makes sense because according to game experience, lee sin is the most popular jungle and thresh is the most popular support.

Other rows are mostly of marksman + support combinations. 

### 2. Frequent enemy combinations
This analysis is a little bit different from the first one for the data pre-processing. Each rows contains two aspects of enemy team, from team1’s view and from team2’s view. And each champion can attend both in team1 and team2. So each row has one basket of size ten. Team2’s champions’ Id would be labeled *.

Items[]
freq
Items[]
freq
497*, 498*
1231
67, 18*
789
497, 498
1131
412, 18
750
18, 412*
872
29*, 18
704
412, 18*
847
67*, 412
693
67*, 18
824
40, 412*
692
18*, 412*
791
67, 412
692

These rows are not all useful. Only those with a plain id and a starred id are frequent enemy combinations.
Take few to validate
18-> “the Yordle Gunner Tristana”
412-> “The Chain Warden Thresh”
It makes sense because according to LOL champion abilities, Thresh can block Tristana when she is jumping.

29*-> “the Plague Rat Twitch”
18-> “the Yordle Gunner Tristana”
It makes sense because these two champions are both marksman. They are fixed to be in bot lane and thus more probable to see each other as enemy.

### 3. Naïve Bayes Classifier Prediction
To predict the winning team, gathering some crucial factors is really important. This analysis collect features of firstTower, firstInhibitor, firstBaron, firstDragon, firstRiftHerald, towerKills(team1’s kill – team 2’s kill), inhibitorkill(team1’s kill – team 2’s kill),dragonkill(team1’s kill – team 2’s kill),riftHeraldKill(team1’s kill – team 2’s kill).
Note, firstTower, firstInhibitor etc these features are not strictly binary team1 or team2. Some matches doesn’t have any tower kill so it’s 0. So here applies multinomial naïve bayes model rather than Bernoulli naïve bayes model.

Winner 0.0 stands for team1, 1.0 stands for team2



Test set accuracy = 0.929852045256745

Accuracy is over 90%. So actually these features are decisive for winning. 

### 4. Combination and Winning rate.
The analysis here uses plain python method to get corresponding winning rate of the first few pairs of frequent teammate from part 1 above.
itemset
Winning rate
Item0
Winning rate
Item1
Winning rate
497, 498
0.525826
497
0.501827
498
0.498024
18, 412
0.456846
18
0.517061
412
0.472466
67, 412
0.485423
67
0.515857
412
0.472466
40, 18
0.588468
40
0.555287
18
0.517061
141, 18
0.492129
141
0.492129
18
0.517061
64, 18
0.504496
64
0.458968
18
0.517061
29, 412
0.481336
29
0.527834
412
0.472466
40, 67
0.595661
40
0.555287
67
0.515857
141, 67
0.503455
141
0.492129
67
0.515857
117, 18
0.504451
117
0.502163
18
0.517061

So we can conclude that, [497, 498], [40, 18], [40, 67] works better than individually. Not all popular combinations are good for winning.
To validate that, 40-> “The Storm’s Fury Janna” and 18->tristana, 67-> “The Night Hunter Vayne” are actually good pairs because Janna can give shield to them.

## IV. Discussion

### 1.Limitation
First, the dataset is not so perfect because matches are from different seasons, which means their champions data are not identical for each match.
Second, for the naïve bayes method, we assume these features are conditionally independent and work for bayes reasoning. But actually it’s not guaranteed. The prediction accuracy of 92% maybe just a coincidence. For example, if player get firstTower kill, he is more likely to get firstInhibitor kill.

### 2.Future Work
This analysis uses naïve bayes for classification (winning team is team1 or team2). There are other classification methods like random forest, multilayer neural network. These methods deserve a try.



























## Appendix

Checklist:
Item
used
Project Report.pdf

game.csv
dataset
frequent_teammate.py

frequent_enemy.py

naïve_bayes.py

combination_winning_rate.py

champion_info.json
Champion id map to champion

Dataset are from https://www.kaggle.com/jaytegge/league-of-legends-data-analysis/data

Before running the code, please run this line:
```
source lolAnalysis/env.sh
```

Report and code are also uploaded in my github: https://github.com/GuHan99/lolAnalysis
