from src.MLP import MLP
import numpy as np
import gymnasium as gym
import time

def run (model: MLP, norm_reward=False):
    env = gym.make("LunarLander-v2", render_mode="human")
    def policy (observation, model):
        s = model.forward(observation)
        action = np.argmax(s)
        return action
    #observation, info = env.reset(seed=42)
    observation, info = env.reset()
    ite = 0
    racum = 0.0
    while True:
        action = policy(observation, model)
        # action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        racum += reward
        # print(reward)

        if terminated or truncated:
            # r = (racum+500) / 700
            # print(racum, racum+500, r)
            # print(racum, r)
            return (racum+500) / 700 if norm_reward else racum

def test():
    ch = np.array(
        [-0.1119248726956276, 0.11172034106210418, 1.0, -0.4518290048469199, 0.036672881361180895, 0.23098951585331717, 0.6229781584644364, -0.12180082955338731, 0.06817701187828042, 0.5442623907235506, -0.08966852367782463, -0.6534182137057888, -0.24026497482981696, -0.1781870473800145, 0.3906635152478246, 0.08430617934006596, -0.17689940724206893, -0.29752893629300714, 0.40409980436676296, -0.3136015955330185, 0.3230185594815511, -0.4362401809108682, 0.46800263107095486, -0.022887718102232914, 0.8209552038794281, 0.09586887853017287, -0.1276921605095718, -0.07656244255790665, 0.15333231113385395, -0.12474101074768357, -0.666485782125479, 0.17301522467031688, -0.22123471887801593, 0.5967958779586001, -0.1064989116397281, 0.27363616168704613, 0.5465460871664758, 0.5133233380044454, -0.11204585289984662, 0.7861569314224854, 0.08063373974858826, -0.05977073952524578, -0.3271234692559092, 0.38830156298126484, 0.3365641246818518, -0.6175343422402926, -0.4746843146766172, -0.6770926804204386, -0.1345148665843292, -0.14585917171195006, -0.4038105438895068, -0.08916503245916504, -0.48700247307456507, -0.038937016528238964, 0.6070846478295756, -0.16295119980590697, 0.16314439107943282, -0.46138398028000316, -0.6016218675188677, 0.13370221905322033, 0.8200291757864815, -0.34635464392041865, -0.1074944663073376, -0.15603558130819378, -0.39082098435780555, 0.7908478824503322, 0.12321727769167476, -0.3385668415087685, -0.5461843275507953, 0.13927383938795393, 0.2299380511390018, -0.20976536434980428, -0.3859057289415917, -0.02759826003077491, 0.4798701491310593, 0.2557388353816838, 0.23318512903520439, -0.3074897855528147, 0.15652132656744877, -0.2474722278337908, 0.39230684967323837, -0.7026943427158847]

    )
    model = MLP(
        layers=[8, 6, 4],
        chromosome=ch
        )
    
    for i in range(10):
        print(run(model))
        time.sleep(1)
    

if __name__ == "__main__":
    test()
