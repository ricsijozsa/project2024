import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


st.set_page_config(page_title="KSH Adatok", page_icon=":bar_chart:", layout="wide")
st.title(":bar_chart: KSH Adatok")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)


st.sidebar.header("Adat Feltöltése")
file = st.sidebar.file_uploader(":file_folder: Töltsön fel egy CSV, TXT, XLSX vagy XLS fájlt", type=["csv", "txt", "xlsx", "xls"])


if file:
    try:
        if file.name.endswith('.csv') or file.name.endswith('.txt'):
            df = pd.read_csv(file, encoding="ISO-8859-1")
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Nem sikerült betölteni a fájlt. Hiba: {str(e)}")
        st.stop()
else:
    st.warning("Nincs fájl feltöltve. Az alapértelmezett adatkészletet használjuk.")
    try:
        df = pd.read_csv("dummy_out.csv", encoding="ISO-8859-1")
    except FileNotFoundError:
        st.error("Az alapértelmezett fájl nem található. Töltsön fel egy fájlt a folytatáshoz.")
        st.stop()


df.columns = [col.strip() for col in df.columns]


st.write("### Adatkészlet Áttekintése")
st.dataframe(df)


if 'Év' not in df.columns:
    st.error("Hiányzik az 'Év' oszlop az adatkészletből.")
    st.stop()

try:
    df['Év'] = pd.to_numeric(df['Év'], errors='coerce')
    df.dropna(subset=['Év'], inplace=True)
    df['Év'] = df['Év'].astype(int)
except Exception as e:
    st.error(f"Az 'Év' oszlop feldolgozása sikertelen. Hiba: {str(e)}")
    st.stop()


st.sidebar.header("Év Szűrő")
year_range = st.sidebar.slider(
    "Válassza ki az Év Skálát:",
    min_value=int(df['Év'].min()),
    max_value=int(df['Év'].max()),
    value=(int(df['Év'].min()), int(df['Év'].max()))
)
filtered_df = df[(df['Év'] >= year_range[0]) & (df['Év'] <= year_range[1])]
st.write("### Szűrt Adatok")
st.dataframe(filtered_df)


columns_to_plot = st.multiselect(
    "Válasszon oszlopokat a vonal diagramhoz:",
    options=df.columns[1:],
    default=df.columns[1:]
)


if columns_to_plot:
    st.write("### Vonal Diagramok")
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


comparison_columns = st.multiselect(
    "Válasszon oszlopokat az összehasonlításhoz:",
    options=df.columns[1:],
    default=df.columns[1:3]
)
if comparison_columns:
    st.write("### Több Oszlop Összehasonlítása")
    fig = px.line(
        filtered_df, x='Év', y=comparison_columns,
        title="Kiválasztott Oszlopok Összehasonlítása",
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


st.write("### Lineáris Regressziós Elemzés")
x_column = st.selectbox("Válassza ki az X-tengelyt (Prediktor):", options=df.columns[1:])
y_column = st.selectbox("Válassza ki az Y-tengelyt (Cél):", options=df.columns[1:])
if x_column and y_column:
    try:
        X = filtered_df[[x_column]].values
        y = filtered_df[y_column].values

        
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

            st.write(f"#### Regresszió Részletei")
            st.write(f"Egyenlet: {y_column} = {model.coef_[0]:.2f} * {x_column} + {model.intercept_:.2f}")
            st.write(f"Átlagos Négyzetes Hiba: {mse:.2f}")
        else:
            st.warning("Nincs elegendő érvényes adatpont a regressziós elemzéshez.")
    except Exception as e:
        st.error(f"Regressziós elemzés hiba: {str(e)}")


st.write("### Összefoglaló")
st.write(f"Adatok {filtered_df['Év'].min()} és {filtered_df['Év'].max()} között, {len(filtered_df)} rekorddal.")
