import irsim

env = irsim.make('verify_case_01.yaml')
for i in range(2000):

    env.step()
    env.render(0.1)
    
    if env.done():
        break

env.end(3)
