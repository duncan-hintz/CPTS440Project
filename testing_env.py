from customEnv import CustomEnv
import numpy as np

def episode():
    #alt: "human", None
    render_mode=None
    parallel_env = CustomEnv(render_mode=render_mode)
    observations,info=parallel_env.reset()

    while parallel_env.agents:        
        #policy
        #Choose an action
        actions = {}
        for agent in parallel_env.agents:
            #This line fixes changes made in poke-env v14 that breaks the old action mask and observations
            observations[agent]=observations[agent]['observation']
            #This uses the sample function, which is random with a mask
            actions[agent]=parallel_env.action_space(agent).sample(mask=observations[agent]["action_mask"])
            
        observations,rewards, terminations, truncations, infos = parallel_env.step(actions)

    for battle_tag, battle in parallel_env.agent1.battles.items():
        print("Finished game ", battle_tag)
    
    parallel_env.close()

if __name__ == "__main__":
    for i in range(1000):
        episode()