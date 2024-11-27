import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


#cim, layout, emoyi + oldal beallitasok.

st.set_page_config(page_title="KSH Project", page_icon=":bar_chart",layout="wide")

st.title(" :bar_chart: KSH Csoport Project")
st.markdown('<style>div.block-container{padding-top:2rem;}</stlye>',unsafe_allow_html=True)

#barki hasznalhatja a dashboardot es feloltheti az alabbi tipusu fileokat, hogy kiolvassa az abbol nyert chartokat, en CSV file al dolgoztam, es beallitottam hogy a dummy_out csv file-t futtassa
#nincs adat feltoltve

fl = st.file_uploader(":file_folder: Adat feltoltese",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
else:
    os.chdir(r"C:\Users\richa\Desktop\project")
    df = pd.read_csv("dummy_out.csv")

# to have clean column names 
df.columns = [col.strip() for col in df.columns]

# display the entire data set which is dummy out csv in our case. 
st.write("### Adat CSV file ")
st.dataframe(df)

# I kept facing the same dif over and over as EV column was not int, therefore could not load dataset 
try:
    df['Év'] = pd.to_numeric(df['Év'], errors='coerce')
    df = df.dropna(subset=['Év'])  # Drop rows where 'Év' could not be converted
    df['Év'] = df['Év'].astype(int)
except KeyError:
    st.error("'Év' nincs a datasetben.")
    st.stop()

# sidebar filter to select range between years for the comparision ( last line should be deleted as its almost 0 in each case )
st.sidebar.header("Ev szurese")
year_range = st.sidebar.slider("Ev skala:", 
                                int(df['Év'].min()), 
                                int(df['Év'].max()), 
                                (int(df['Év'].min()), int(df['Év'].max())))

# to filter on the data based on the selected year range ( again, 2023 will be removed from file )
filtered_df = df[(df['Év'] >= year_range[0]) & (df['Év'] <= year_range[1])]

# to show the filtered data 
st.write("### Megszurt adat")
st.dataframe(filtered_df)

# to select different column for comparision ( not final version, looks cheap)
columns_to_plot = st.multiselect("Valassz erteket az adat leolvasashoz:", df.columns[1:], default=df.columns[1:])

# line chart 
if columns_to_plot:
    st.write("### Vonal Diagram")
    for column in columns_to_plot:
        
        fig = px.line(
            filtered_df, 
            x='Év', 
            y=column, 
            title=f'{column} Evek soran', 
            labels={'Év': 'Targy Ev', column: column},
            markers=True
        )
        # layout for the line charts ( this should be smaller, as it covers almost the entire screan - to modify )
        fig.update_layout(
            title_font_size=18,
            title_x=0.5,
            xaxis_title="Ev",
            yaxis_title=column,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Valassz legalabb egyet ay osszehasonlitashoz.")

# I left some space to summarize what we read out from the charts ( also linear regression should be applied somehwere here - prolly before the summary ? ( team discuss with the team)
st.write("### Osszesito")
st.write(f"Adatok eves lebontasban {filtered_df['Év'].min()} to {filtered_df['Év'].max()}.")



# charts to compare different data columns - into 1 chart 
st.write("### Hasonlitson ossze adatokat egy charton")

# whoever using it, can select ftom the columns , so compariosn can be individual 
comparison_columns = st.multiselect(
    "Valasszon erteket az osszehasonlitashoz:",
    df.columns[1:],  # remove EV column,
    default=df.columns[1:3]
)

if comparison_columns:
    
    fig, ax = plt.subplots(figsize=(4, 3))

    for column in comparison_columns:
        ax.plot(filtered_df['Év'], filtered_df[column], marker='o', linestyle='-', label=column)

    # chart layout - ( numbers should be exchanged due to the fact the chartes are taking up the entire page ??? )
    ax.set_title("Adatok  összehasonlítása", fontsize=5)
    ax.set_xlabel("Ev", fontsize=1)
    ax.set_ylabel("Ertek", fontsize=1)
    ax.legend(title="?", fontsize=3, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)

    
    st.pyplot(fig)
else:
    st.write("Kérjük, válasszon legalább egy oszlopot az összehasonlításhoz!")
