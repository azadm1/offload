import numpy as np
import pandas as pd
from EnergyEnv import EnergyEnv
from agent import DQNAgent
import time

if __name__ == "__main__":

    # intial prep
    len = 4000
    episodes = 2000
    batch_size = 32
    data_index = ['task%d' % (i + 1) for i in range(len)]
    data_list = pd.Series(np.random.randint(10, 31, size=len), data_index)

    results = []
    timestamp = time.strftime('%Y%m%d%H%M')

    env = EnergyEnv(data_list)

    state_size =env.action_space.n
    action_size = env.observation_space.n
    agent = DQNAgent(state_size,action_size)

    for e in range(episodes):
        state = env._reset()

        for time in range(env.n_step):
            action = agent.act(state)
            next_state,reward,done,info = env._step(action)
            agent.remember(state,action,reward,next_state,done)
            state = next_state

            if done:
                print("episode: {}/{}, episode end value: {} {}".format(
                    e + 1, episodes, info['cur_val'],state))
                results.append(info['cur_val'])  # append episode end portfolio value
                break

            if  agent.cur_size > batch_size:
                agent.replay(batch_size)

    if (e+10) % 10 == 0 :
        agent.save('weights/{}-dqn.h5'.format(timestamp))





