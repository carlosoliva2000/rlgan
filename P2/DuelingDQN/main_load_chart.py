# load the csv
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('lunar_lander_dueling_rewards_softupdate.csv')
df = df[['rewards', 'steps']]
df['Score'] = df['rewards']

# make an average of the last 100 rewards
df['Average Score'] = df['Score'].rolling(window=100).mean()
df['Solved Threshold'] = 200


plt.plot('Score', data=df, marker='', color='blue', linewidth=2, label='Score')
plt.plot('Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed', label='AVG score')
plt.plot('Solved Threshold', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
         label='Umbral resuelto')
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.legend()
plt.show()


# get the average, median and average steps of the last 10 episodes
print(df['Score'].tail(10).mean())
print(df['Score'].tail(10).median())
print(df['steps'].tail(10).mean())
