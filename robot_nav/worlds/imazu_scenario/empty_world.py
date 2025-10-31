import irsim

env = irsim.make('imazu_case_22.yaml')
for i in range(1000):

    env.step()
    env.render(0.01)
    
    if env.done():
        break

env.end(3)
