from src.data.loader import load
from src.analysis.visualise import visualise
from src.setup.config import SOURCES, RTE_STEPS
from src.setup.constant import MAPPING_STEPS

class EnergyMarket:

    def __init__(self):
        self.data = None
        self.sources = SOURCES
        self.steps =[MAPPING_STEPS[step] for step in RTE_STEPS]


    def initialize(self):
        self.data = load(steps=self.steps)

    def visualize(self):
        visualise(self.data, self.sources, self.steps)