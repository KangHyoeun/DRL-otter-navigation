import irsim

env = irsim.make('imazu_case_11.yaml')
for i in range(500):

    env.step()
    env.render(0.01)
    
    if env.done():
        break

env.end(3)
