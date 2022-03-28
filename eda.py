import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time 
import re 
import json
from streamlit_lottie import st_lottie
#from feateng import *

### INDICE

sidebar = st.sidebar
sidebar.markdown('''
# Indice
--- 
**1.** [Conta dei Tickets per variabile](#1)

**2.** [Aggrega una variabile e vedi la ditribuzione dei voti](#2)

**3.** [Troviamo gli operatori più efficienti](#3)

**4.** [Troviamo gli operatori meno efficienti](#4)

**5.** [Quanto tempo passa tra l'apertura di un ticket e la sua chiusura?](#5)

**6.** [Aggrega per voto e per prodotto](#6)

**7.** [Download del dataset con le nuove variabili e/o filtrato](#7)

--- 
''', unsafe_allow_html=True)


colo1, colo2 = st.columns((1,1))
with colo1:
    st.image("img/logo-infocert.png",width=300)
    st.caption('Pera Lorenzo')

with colo2:
    with open("data/service.json", "r") as f:
        data = json.load(f)
    st_lottie(data,width=300)
    


st.title ("VTE_APP - EDA (Exploratory Data Analysis)")
# Descrizione dell'applicazione
st.markdown("## Con questa applicazione sarà possibile esplorare e **conoscere meglio i nostri dati**\
    \nNel dettaglio sarà possibile:\n- Ottenere dei **grafici interattivi** in base ai filtri che si trovano a sinistra\
        \n- Visualizzare **la distribuzioni dei voti al variare dei filtri**\n- Analizzare il **$Δ$ tra apertura/chiusura tickets**\n- Trovare i **migliori/peggiori operatori**")


eda_file = st.file_uploader('Carica un file excel scaricato da VTE (un file di default è stato fornito)\n\
    ATTENZIONE: il file che verrà caricato deve avere le stesse colonne di quello di default (scritte nello stesso modo)')

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def load1(file):
    return pd.read_excel(file)

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def load2():
    return pd.read_excel('data\FEEDBACK_ex.xlsx')

if eda_file is not None:
    vte = load1(file=eda_file)
else:
    vte = load2()    

st.text('Anteprima del file caricato:')
st.write(vte.head())    
st.text("")
st.text("")
st.text("")
#####   CREO NUOVE VARIABILI
vte['data_open'] = vte.createdtime.dt.date
vte['giorno_open'] = vte.createdtime.dt.day_name()
vte['ora_open'] = vte.createdtime.dt.hour
# chiusura
vte['data_close'] = vte['closing_time (TT VTE)'].dt.date
vte['giorno_close'] = vte['closing_time (TT VTE)'].dt.day_name()
vte['ora_close'] = vte['closing_time (TT VTE)'].dt.hour
vte.head(3)
# tempo trascorso tra open e close
vte['tempo_chiusura'] = (vte['closing_time (TT VTE)']-vte.createdtime)
vte['tempo_chiusura_H'] = vte.tempo_chiusura / pd.Timedelta(hours=1) # in ore
vte['tempo_chiusura'] = vte['tempo_chiusura'].dt.days
# metodo interquantile per trovare outliers
q1,q3 = np.quantile(vte.tempo_chiusura_H.fillna(0),[0.25,0.75])
IQR = q3-q1
up_outlier = q3+(1.5*IQR)
dw_outlier = q1-(1.5*IQR)

vte['tempi_troppo_lunghi'] = np.where(vte.tempo_chiusura_H>up_outlier,1,0)

# merge 
vte_full = vte.merge(vte.groupby('operatore').gruppoFamiglia.apply(list).to_frame('PRODOTTI').\
assign(prods_operatore=lambda df: df.PRODOTTI.apply(lambda x: set(x))).\
assign(Tot_tickets_operatore=lambda df: df.PRODOTTI.str.len())[['prods_operatore','Tot_tickets_operatore']],
          left_on='operatore',right_index=True).assign(n_prods_operatore=lambda df:df.prods_operatore.str.len()).\
assign(prods_operatore=lambda df: df['prods_operatore'].str.join(','))

####### FUNZIONI
def contavalori(data,colonna):
    return data[colonna].value_counts().to_frame(colonna.upper()).\
assign(NORMALIZZATI=data[colonna].value_counts(normalize=True).mul(100).round(2)).\
assign(CUMULATIVA=lambda df: df.NORMALIZZATI.cumsum())

def raggruppa(data,col1,col2):
    return data.groupby([col1,col2])[col2].count().unstack().fillna(0).astype(int).\
assign(TOT=lambda df: df.sum(axis=1)).pipe(lambda df: df.join(df.drop('TOT',axis=1).astype(int).\
                                                              div(df.TOT,axis=0).round(3).mul(100),
                                                              lsuffix='_abs',
                                                              rsuffix='_pct'))


def cerca_anni(year):
    x = re.search(r"(19[0-9]{2})|([4-9]{2})", year)
    if x :
        return(x.group())

#################################     CREO LE MASCHERE....SIDEBA + SIDEBAR COLOR




prodotti = sidebar.multiselect('Seleziona il prodotto',vte_full.gruppoFamiglia.unique(),
vte_full.gruppoFamiglia.unique())
mask_prodotti = vte_full.gruppoFamiglia.isin(prodotti)

voti = sidebar.multiselect('Seleziona il voto',vte_full.voto_feedback.unique(),
vte_full.voto_feedback.unique())
mask_voti = vte_full.voto_feedback.isin(voti)
sidebar.markdown('Per trovare gli outliers ho usato metodo [IQR Inter Quantile](https://en.wikipedia.org/wiki/Interquartile_range)')
too_long = sidebar.multiselect('Seleziona 1 per eliminare gli outliers (tempi di chiusura oltre le 100 ore circa (4 gg)) '
,vte_full.tempi_troppo_lunghi.unique(),
vte_full.tempi_troppo_lunghi.unique())
mask_too_long = vte_full.tempi_troppo_lunghi.isin(too_long)

vte_full_masked = vte_full[mask_prodotti&mask_voti&mask_too_long].copy()



###### CONTA VALORI 
st.header('Conta',anchor='1')
variabile_conta_valori = st.selectbox('Seleziona una variabile per contare il numero di Tickets (Esempio: operatore)'
,vte_full.columns.tolist(),index=4)
st.write(contavalori(vte_full_masked,variabile_conta_valori))
st.text("")
st.text("")
st.text("")

#### RAGGRUPPA
st.header("Aggrega",anchor='2')
st.markdown('Conta di una variabile in dipendenza della distribuzione dei voti sui feedback\n\
- anche in questo caso potrai selezionare una variabile a tua scelta\n\
- otterrai una tabella con i valori assoluti, il totale, e i valori normalizzati\n\
- grazie al gradiente del colore è possibile trovare subito valori interessanti\n\
- non potrai (ovviamente) filtrare per voto!')
col1 = st.selectbox('Seleziona la variabile da aggregare',vte_full.columns.tolist(),index=1)
st.write(raggruppa(vte_full[mask_prodotti&mask_too_long],col1,'voto_feedback').\
style.background_gradient(axis=1,subset=raggruppa(vte_full,col1,'voto_feedback').columns[:5]).\
background_gradient(axis=1,subset=raggruppa(vte_full,col1,'voto_feedback').columns[6:],cmap='YlOrRd').\
background_gradient(axis=0,subset=raggruppa(vte_full,col1,'voto_feedback').columns[5],cmap='Greens'))

####### migliori operatori e peggiori 
st.text("")
st.text("")
st.text("")
st.header("Operatori migliori",anchor='3')
st.markdown('Migliori operatori (si può filtrare per prodotto e presenza di outliers) \n\
### per trovare i migliori operatori:\n')
st.markdown('- ho selezionato operatori con  totale tickets > 10\n\
- dove percentuale di punteggio 5 sia **maggiore** a quella di punteggio 1 e 2')

OP = raggruppa(vte_full[mask_prodotti&mask_too_long],'operatore','voto_feedback').sort_values('TOT',ascending=False)
OP.sort_values('5_pct',ascending=False).query('TOT>50').query('`5_pct`>50')
mask1 = OP.apply(lambda df: df['5_pct']>df['1_pct'],axis=1)
mask2 = OP.apply(lambda df: df['5_pct']>df['2_pct'],axis=1)

st.write(OP[mask1&mask2].sort_values('5_pct',ascending=False).query('TOT>10').\
style.background_gradient(axis=1,subset=OP.columns[:5]).\
background_gradient(axis=1,subset=OP.columns[6:],cmap='YlOrRd').\
background_gradient(axis=0,subset=OP.columns[5],cmap='Greens'))

TOPPP = OP[mask1&mask2].sort_values('5_pct',ascending=False).query('TOT>10').\
style.background_gradient(axis=1,subset=OP.columns[:5]).\
background_gradient(axis=1,subset=OP.columns[6:],cmap='YlOrRd').\
background_gradient(axis=0,subset=OP.columns[5],cmap='Greens').index.tolist()
st.write(TOPPP)

st.write(f"Con i migliori abbiamo circa {vte_full[vte_full.operatore.isin(TOPPP)].shape[0]} tickets")


st.text("")
st.text("")
st.text("")

st.header("Operatori peggiori",anchor='4')
st.markdown('Peggiori operatori (si può filtrare per prodotto e presenza di outliers) \n\
### per trovare i peggiori operatori:\n')
st.markdown('- ho selezionato operatori con  totale tickets > 50, percentuale punteggio 1 > 50%\n\
- a quelli sopra ho aggiunto gli operatori con percentuale di punteggio 5 sia **minore** a quella di punteggio 1 e 2')

OP_down = raggruppa(vte_full[mask_prodotti&mask_too_long],'operatore','voto_feedback')
#OP_down.query('TOT>50').query('`1_pct`>50').sort_values(['1_pct','5_pct'],ascending=[False,True])
mask3 = OP_down.apply(lambda df: df['5_pct']<df['1_pct'],axis=1)
mask4 = OP_down.apply(lambda df: df['5_pct']<df['2_pct'],axis=1)

st.write(OP_down.query('TOT>50').query('`1_pct`>50').sort_values(['1_pct','5_pct'],ascending=[False,True]).append(
OP_down[mask3&mask4].sort_values(['1_pct','5_pct'],ascending=[False,True])).\
style.background_gradient(axis=1,subset=OP_down.columns[:5]).\
background_gradient(axis=1,subset=OP_down.columns[6:],cmap='YlOrRd').\
background_gradient(axis=0,subset=OP_down.columns[5],cmap='Greens'))

BADDD = OP_down.query('TOT>50').query('`1_pct`>50').sort_values(['1_pct','5_pct'],ascending=[False,True]).append(
OP_down[mask3&mask4].sort_values(['1_pct','5_pct'],ascending=[False,True])).\
style.background_gradient(axis=1,subset=OP_down.columns[:5]).\
background_gradient(axis=1,subset=OP_down.columns[6:],cmap='YlOrRd').\
background_gradient(axis=0,subset=OP_down.columns[5],cmap='Greens').index.tolist()
st.write(BADDD)

st.write(f"Con i peggiori abbiamo circa {vte_full[vte_full.operatore.isin(BADDD)].shape[0]} tickets")

st.text("")
st.text("")
st.text("")



######################### GRAFICO LINEA
open_=contavalori(vte_full_masked,'ora_open').ORA_OPEN.sort_index()
close=contavalori(vte_full_masked,'ora_close').ORA_CLOSE.sort_index()
newdata = open_.to_frame('open').join(close.to_frame('close').fillna(0))
#fig = px.line(data, x=open_.index,y=data[['open','close']], title='Life expectancy in Canada')
#fig.show()
figPLO = go.Figure()
figPLO.add_trace(go.Scatter(x=newdata.index, y=newdata.open,
                    mode='lines+markers',line=dict(color='#ffb300'),
                    name='open'))
figPLO.add_trace(go.Scatter(x=newdata.index, y=newdata.close.fillna(0),
                    mode='lines+markers',line=dict(color='#023047'),
                    name='close'))

# Edit the layout
figPLO.update_layout(title='Conta degli orari di apertura/chiusura tickets', 
                  title_xanchor='left',title_font_family='Arial',
                   xaxis_title='Ore del giorno',title_font_size=26,
                   yaxis_title='Numero di Tickets',
                   
                   xaxis=dict(
        tickmode = 'array',
        tickvals = [*range(0,25)],
        ticktext = [*range(0,25)],                        
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        title_font_family='Arial',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    yaxis=dict(
        linecolor='rgb(204, 204, 204)',
        linewidth=2,        
        showgrid=True,
        zeroline=True,
        showline=True,
        showticklabels=True,
        ticks='outside',
                tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    autosize=True,
    margin=dict(
        autoexpand=True,
        l=100,
        r=20,
        t=110,
    ),
    showlegend=True,legend=dict(title='Legenda',x=0.06,
                                title_font_family='Arial',
                                font=dict(family='Arial')),

    plot_bgcolor='white'
                 )



################################ GRAFICO DATE

figPLO2 = go.Figure()
figPLO2.add_trace(go.Scatter(x=contavalori(vte_full_masked,'data_open').DATA_OPEN.sort_index().index,
                         y=contavalori(vte_full_masked,'data_open').DATA_OPEN.sort_index(),
                    mode='lines+markers',line=dict(color='#ffb300'),
                    name='Storico open'))
figPLO2.add_trace(go.Scatter(x=contavalori(vte_full_masked,'data_close').DATA_CLOSE.sort_index().index, 
                         y=contavalori(vte_full_masked,'data_close').DATA_CLOSE.sort_index().fillna(0),
                    mode='lines+markers',line=dict(color='#023047'),
                    name='Storico close'))

# Edit the layout
figPLO2.update_layout(title='Conta apertura/chiusura giornaliera dei tickets', 
                  title_xanchor='left',title_font_family='Arial',
                   xaxis_title='Giorni',title_font_size=26,
                   yaxis_title='Numero di Tickets',
                   
                   xaxis=dict(
        tickmode = 'array',
        #tickvals = [*range(0,25)],
        #ticktext = [*range(0,25)],                 
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        title_font_family='Arial',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    yaxis=dict(
        linecolor='rgb(204, 204, 204)',
        linewidth=2,        
        showgrid=True,
        zeroline=True,
        showline=True,
        showticklabels=True,
        ticks='outside',
                tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    autosize=True,
    margin=dict(
        autoexpand=True,
        l=100,
        r=20,
        t=110,
    ),
    showlegend=True,legend=dict(title='Legenda',x=0.06,
                                title_font_family='Arial',
                                font=dict(family='Arial')),

    plot_bgcolor='white'
                 )



container = st.container()
container.header("Il fattore tempo",anchor='5')
container.markdown('### Analisi delle variabili temporali')
container.write('\n\n')
GRAF = container.selectbox('Seleziona un grafico',["Analisi dell' orario",'Analisi delle date'])
if GRAF == "Analisi dell' orario":
    container.plotly_chart(figPLO)
    container.markdown("Possiamo notare che molti tickets vengono aperti dopo l'orario di lavoro **in un numero superiore rispetto a quelli chiusi**")
else :
    container.plotly_chart(figPLO2)     
    container.markdown("Ad eccezione di alcuni sporadici picchi, la linea della chiusura Tickets sembra sovrapporsi a quella dell'apertura, **indice di un'assistenza tempestiva**")
    container.markdown("\nPossiamo avere una migliore visualizzazione della distribuzione utilizzando un boxplot")
    # altro grafico pyplot
    container.markdown("\nInfatti gran parte dei tickets vengono chiusi nel giro di 5 ore\n *Outliers rimossi*")
    st.markdown('### Analisi outliers')
    st.markdown('Applicando questi filtri hai i seguenti outliers')
    st.write(vte_full[mask_prodotti&mask_voti][vte_full.tempi_troppo_lunghi==1].tempo_chiusura_H.to_frame('outliers').describe())
    fig3, ax3 = plt.subplots()
    ax4 = ax3.twinx()     
    # histogram
    vte_full[mask_prodotti&mask_voti][vte_full.tempi_troppo_lunghi==0].\
        tempo_chiusura_H.plot.hist(figsize=(10,3),alpha=1.0,ax=ax3,bins=20,density=False,color = "#023047")
    plt.xticks(np.arange(0,125,5),np.arange(0,125,5))
    # boxplot
    vte_full[mask_prodotti&mask_voti][vte_full.tempi_troppo_lunghi==0].\
        tempo_chiusura_H.plot.box(figsize=(10,3),
                 color=dict(boxes="#ffb300", whiskers="#ffb300", medians="#ffb300", caps="#ffb300"),
             boxprops=dict(linestyle='-', linewidth=3.5,color='#ffb300'),
             flierprops=dict(linestyle='-', linewidth=3.5,color='#ffb300'),
             medianprops=dict(linestyle='-', linewidth=3.5,color='#ffb300'),
             whiskerprops=dict(linestyle='-', linewidth=3.5,color='#ffb300'),
             capprops=dict(linestyle='-', linewidth=3.5,color='#ffb300'),    
        showmeans=True,vert=False,ax=ax4)#,color = "#ffb300")
   
    ax4.set_xlabel('Numero di ore passate tra apertura e chiusura di un ticket',loc='left')
    ax3.set_ylabel('Conta dei tickets')
    plt.title('Distribuzione dei tempi di chiusura tickets (in ore, outliers rimossi)',loc='left',weight='bold')
    st.pyplot(fig3)
st.text('')   
st.text('')
st.text('')


###################################  BAR CHART CATEGORICA
st.header("Questo grafico contiene molte informazioni utili per valutare la distribuzione dei voti",
anchor=6)
st.markdown(
"""
**Abbiamo un raggruppamento per prodotto**
- Una distribuzione percentuale dei voti all'interno del prodotto\n\
- Passando con il mouse possiamo ottenere altre informazioni:
    1. percentuale sul prodotto

    2. totale tickets in valore assoluto

    3. percentuale sul voto di riferimento

    4. totale prodotto

    5. totale voto

""")

piv2 = vte_full.pivot_table(index=['gruppoFamiglia','voto_feedback'],values='Tot_tickets_operatore',
                     aggfunc='count')
prova4 = piv2.merge(piv2.groupby(level=[0]).sum(),left_index=True,right_index=True,suffixes=('','to_drop')).\
merge(piv2.groupby(level=[1]).sum(),left_index=True,right_index=True,suffixes=('','to_drop2')).\
assign(percentuale_sul_prodotto=lambda df: df.Tot_tickets_operatore.div(df.Tot_tickets_operatoreto_drop).mul(100).round(2)).\
assign(percentuale_sul_voto=lambda df: df.Tot_tickets_operatore.div(df.Tot_tickets_operatoreto_drop2).mul(100).round(2)).\
reset_index()
#drop(['Tot_tickets_operatoreto_drop','Tot_tickets_operatoreto_drop2'],axis=1).reset_index()
prova4.columns = ['PRODOTTO','VOTO FEEDBACK','TOTALE_TICKETS','TOT PRODOTTO','TOT VOTO','% SUL PRODOTTO','% SUL VOTO']


figPROVA4 = px.bar(prova4, x='PRODOTTO',y='% SUL PRODOTTO',title="Distribuzione dei voti divisi per prodotto".upper(),
             #color_continuous_scale=["#0d3b66","#faf0ca","#f4d35e","#ee964b","#61a5c2"],
             color_continuous_scale=[(0.00, "#023047"),   (0.20, "#023047"),
                                     (0.20, "#faf0ca"), (0.40, "#faf0ca"),
                                     (0.40, "#f4d35e"), (0.60, "#f4d35e"),
                                     (0.60, "#ee964b"), (0.80, "#ee964b"),                                     
                                     (0.80, "#61a5c2"),  (1.00, "#61a5c2")],
                   color="VOTO FEEDBACK",barmode='group',hover_data=['TOTALE_TICKETS','% SUL VOTO',
                                                                     'TOT PRODOTTO','TOT VOTO'],)

figPROVA4.update_xaxes(
                        
            showline=True,
            showgrid=False,
            showticklabels=True)


figPROVA4.update_layout(plot_bgcolor='white',
                  margin=dict(autoexpand=True),
                 coloraxis_colorbar=dict(
    title="VOTO FEEDBACK",
    tickvals=[1,2,3,4,5],
    ticktext=["1", "2", "3", "4", "5"],
))

st.plotly_chart(figPROVA4)
st.text("")
st.text("")
st.text("")

############## più dettaglio


# prova download

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')


csv = convert_df(vte_full_masked)

st.header("Download",anchor=7)
st.markdown('### GUIDA AL DOWNLOAD: \n1. Inserisci un nome senza spazi,\
    \n2. premi ENTER\n3. Clicca su "Press to Download"')
nome_file = st.text_input('Se vuoi puoi scaricare il dataset con le nuove variabili: inserisci un nome')

st.download_button(
"Press to Download",
csv,
f"{nome_file}.csv",
"text/csv",
key='download-csv'
)

st.write("### Anteprima del dataset che andrai ad scaricare.")
st.text(f"Il dataseta avrà {vte_full_masked.shape[0]} righe e {vte_full_masked.shape[1]} colonne")
st.dataframe(vte_full_masked.sample(50))
st.text("")
st.text("")
st.text("")







              