from customEnv import CustomEnv

def episode():
    #alt: "human"
    render_mode=None
    parallel_env = CustomEnv(render_mode=render_mode)
    observations,info=parallel_env.reset()

    while parallel_env.agents:
        #policy
        #Choose an action
        actions = {}
        for agent in parallel_env.agents:
            #This uses the sample function, which is random with a mask
            actions[agent]=parallel_env.action_space(agent).sample(mask=observations[agent]["action_mask"])
            
        observations,rewards, terminations, truncations, infos = parallel_env.step(actions)

    for battle_tag, battle in parallel_env.agent1.battles.items():
        print("Finished game ", battle_tag)
    
    parallel_env.close()

if __name__ == "__main__":
    for i in range(10):
        episode()