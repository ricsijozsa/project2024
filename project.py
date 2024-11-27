import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Oldal beallitasa 
st.set_page_config(page_title="KSH Adatok", page_icon=":bar_chart:", layout="wide")
st.title(":bar_chart: KSH Adatok")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

# Fajl feltoltese, ha nincs semmi akkor dummy file az alapertelmezett ( nem hiszem h mukodik massal, mert a dummy csv re korlatoztam a kodokat)
st.sidebar.header("Adat Feltöltése")
file = st.sidebar.file_uploader(":file_folder: Töltsön fel egy CSV, TXT, XLSX vagy XLS fájlt", type=["csv", "txt", "xlsx", "xls"])

if file:
    df = pd.read_csv(file, encoding="ISO-8859-1")
else:
    st.warning("Nincs fájl feltöltve. Az alapértelmezett adatkészletet használjuk.")
    df = pd.read_csv("dummy_out.csv")  # Cserélje ki egy tartalék fájl elérési útra, ha szükséges

df.columns = [col.strip() for col in df.columns]
st.write("### Adatkészlet Áttekintése")
st.dataframe(df)

# ellenorzes hogy az ev oszlop numerikus legyen 
try:
    df['Év'] = pd.to_numeric(df['Év'], errors='coerce')
    df.dropna(subset=['Év'], inplace=True)
    df['Év'] = df['Év'].astype(int)
except KeyError:
    st.error("Hiányzik az 'Év' oszlop az adatkészletből.")
    st.stop()

# Csuszka az Ev intervallum kivalasztasahoz
st.sidebar.header("Év Szűrő")
year_range = st.sidebar.slider(
    "Válassza ki az Év Skálát:",
    min_value=int(df['Év'].min()),
    max_value=int(df['Év'].max()),
    value=(int(df['Év'].min()), int(df['Év'].max()))
)
filtered_df = df[(df['Év'] >= year_range[0]) & (df['Év'] <= year_range[1])]
st.write("### Szűrt adatok")
st.dataframe(filtered_df)

# Oszlop Választó Vonal Diagramhoz
columns_to_plot = st.multiselect(
    "Válasszon oszlopokat a vonal diagramhoz:",
    options=df.columns[1:],
    default=df.columns[1:]
)

# Vonal diagram 
if columns_to_plot:
    st.write("### Vonal diagramok")
    for column in columns_to_plot:
        fig = px.line(
            filtered_df, x='Év', y=column, 
            title=f"{column} az Évek Során",
            labels={'Év': 'Év', column: column},
            markers=True
        )
        fig.update_layout(
            title_font_size=18,
            xaxis_title="Év",
            yaxis_title=column,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Válasszon legalább egy oszlopot a diagramok generálásához.")

# Tobb oszlopokat osszehasonlito diagram 
comparison_columns = st.multiselect(
    "Válasszon oszlopokat az összehasonlításhoz:",
    options=df.columns[1:],
    default=df.columns[1:3]
)
if comparison_columns:
    st.write("### Több oszlop összehasonlítás")
    fig = px.line(
        filtered_df, x='Év', y=comparison_columns,
        title="Kiválasztott oszlopok összehasonlítása",
        labels={'Év': 'Év', 'variable': 'Változó', 'value': 'Érték'},
        markers=True
    )
    fig.update_layout(
        title_font_size=18,
        xaxis_title="Év",
        yaxis_title="Értékek",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Válasszon legalább egy oszlopot az összehasonlításhoz.")

# Linearis regresszio elemzese
st.write("### Lineáris regressziós elemzés")
x_column = st.selectbox("Válassza ki az X-tengelyt (Prediktor):", options=df.columns[1:])
y_column = st.selectbox("Válassza ki az Y-tengelyt (Cél):", options=df.columns[1:])
if x_column and y_column:
    X = filtered_df[[x_column]].values
    y = filtered_df[y_column].values

    # Ertek kezeles X vagy Y eseteben
    valid_indices = ~np.isnan(X).flatten() & ~np.isnan(y)
    X = X[valid_indices]
    y = y[valid_indices]

    if len(X) > 0 and len(y) > 0:
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)

        fig = px.scatter(
            filtered_df.dropna(subset=[x_column, y_column]), x=x_column, y=y_column,
            title=f"Regresszió: {y_column} vs {x_column}",
            labels={x_column: x_column, y_column: y_column}
        )
        fig.add_scatter(x=X.flatten(), y=y_pred, mode='lines', name='Regressziós vonal')
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.write(f"#### Regresszió részletei")
        st.write(f"Egyenlet: {y_column} = {model.coef_[0]:.2f} * {x_column} + {model.intercept_:.2f}")
        st.write(f"Átlagos négyzetes hiba: {mse:.2f}")
    else:
        st.warning("Nincs elegendő érvényes adatpont a regressziós elemzéshez.")

# Összefoglaló Szekció
st.write("### Összefoglaló")
st.write(f"Adatok {filtered_df['Év'].min()} és {filtered_df['Év'].max()} között, {len(filtered_df)} rekorddal.")
