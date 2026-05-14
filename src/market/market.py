from src.data.loader import load_RTE,load_price
from src.analysis.visualise import visualise
from src.models.princing_from_mix import pricing_model,predict_price_from_mix
from src.setup.config import SOURCES, RTE_STEPS,PRINCING_MODEL_TYPE
from src.setup.constant import MAPPING_STEPS

class EnergyMarket:

    def __init__(self):
        self.data = None
        self.sources = SOURCES
        self.steps =[MAPPING_STEPS[step] for step in RTE_STEPS]
        self.pricing_model_type = PRINCING_MODEL_TYPE

    def initialize(self):
        self.data_RTE = load_RTE(steps=self.steps)
        self.data_price = load_price()

    def visualize(self):
        visualise(self.data_RTE, self.sources, self.steps)

    def price_from_mix_learning(self):
        self.pricing_model, self.feature_cols = pricing_model(self.data_RTE,self.data_price,self.pricing_model_type)

    def predict_from_mix(self,energy_consumption):
        result = predict_price_from_mix(self.pricing_model, self.feature_cols, energy_consumption)
        print(f"Pour cette disposition, le prix de l'électricité est {result}")