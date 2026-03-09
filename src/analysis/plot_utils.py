import matplotlib.pyplot as plt 
import pandas as pd 

def plot_energy_step(df: pd.DataFrame, energy: list, step: str):
    
    df_step = df.groupby(step)[energy].sum().reset_index()

    for col_name in energy:
        plt.plot(df_step[step], df_step[col_name], label=col_name)

    plt.xlabel("Date")
    plt.ylabel("Puissance électrique fournie (en kWh)")
    plt.title(f"Puissance électrique fournie par énergie au pas : {step}")
    plt.legend()
    plt.show()