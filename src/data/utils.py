import pandas as pd 



def clean_RTE_data(df : pd.DataFrame , Sources : list):
    
    # Renaming columns names because of accent 

    df=df.rename(columns={"PÈrimËtre":"pays","PrÈvision J-1":"Prevision J-1","PrÈvision J":"Prevision J","NuclÈaire":"Nucleaire"})

    # Working only in France 

    df=df[df["pays"]=='France']
    
    # Keeping only useful columns 

    df= df[['Date', 'Heures', 'Consommation', 'Prevision J-1',
       'Prevision J', 'Fioul', 'Charbon', 'Gaz', 'Nucleaire', 'Eolien',
       'Solaire']]
    # Converting datetime ti right module 

    df["Date"] = pd.to_datetime(df["Date"].astype(str) + " " +  df["Heures"].astype(str))
    df = df.drop(columns="Heures")

    # On garde seulement les données viables (toutes les semi heures) 

    df = df[df["Date"].dt.minute.isin([0,30])].reset_index(drop=True)
    
    return(df)

