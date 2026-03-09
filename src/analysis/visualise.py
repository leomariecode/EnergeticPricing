import src.analysis.plot_utils as putils

def visualise(data, sources, steps):
    for step in steps:
        for source in sources:
            putils.plot_energy_step(data, [source], step)
        putils.plot_energy_step(data, sources, step)