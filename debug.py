from src.market.market import EnergyMarket
from src.market.market_V2 import EnergyMarketV2

START = "2024-01-01"
END = "2026-05-14"
MODEL = "Simple"
def run():

    # Création du marché
    ## Start END : "YYYY-MM-DD".
    market = EnergyMarketV2(start=START, end=END)

    # Loading et cleaning des datas

    market.initialize()
    # Apprentissage du modèle pour le price
    ## Model Types supportés : "Simple" (Ridge + StandardScaler), "RandomForest", "GradientBoosting"
    market.learn(model_type=MODEL)
if __name__ == "__main__":
    run()