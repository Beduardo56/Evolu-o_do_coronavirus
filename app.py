import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from models import models, optimizer

@st.cache(persist=True)
def load_data() -> (np.array, np.array, np.array):
    ydata = np.array([      1,       1,       1,       2,       2,       2,       2,
             3,       7,      13,      19,      25,      25,      34,
            52,      77,      98,     121,     200,     234,     291,
           428,     621,     904,    1128,    1546,    1891,    2201,
          2433,    2915,    3417,    3903,    4256,    4579,    5717,
          6834,    7910,    9056,   10278,   11130,   12056,   13717,
         15927,   17857,   19638,   20727,   22169,   23430,   25262,
         28320,   30425,   33682,   36599,   38654,   40581,   43079,
         45757,   49492,   52995,   58509,   61888,   66501,   71886,
         78162,   85380,   91299,   96396,  101147,  107780,  114715,
        125218,  135106,  145328,  155939,  162699,  168331,  177589,
        188974,  202918,  218223,  233142,  241080,  254220,  271628,
        291579,  310087,  330890,  347398,  363211,  374898,  391222,
        411821,  438238,  465166,  498440,  514200,  526447,  555383,
        584016,  614941,  645771,  672846,  691758,  707412,  739503,
        772416,  802828,  828810,  850514,  867624,  888271,  923189,
        955377,  978142, 1032913, 1067579, 1085038, 1106470, 1145906,
       1188631])
    xdata = np.arange(len(ydata))
    return (xdata, ydata)

def sir_model_dashboard(ydata, xdata):
    length = len(ydata)
    a0 = st.number_input('Selecione o numero inicial acumulado de casos de covid')
    i0 = st.number_input('Selecione o numero inicial de infectados')
    s0 = st.number_input('Selecione o numero inicial de pessoas saudaveis')
    alpha = st.number_input('Taxa de recuperação')
    gama = st.number_input('Taxa de transmissão')
    ycalc = models.sir_model(length,  a0, s0, i0, gama, alpha)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xdata, y=ydata,
                        mode='lines',
                        name='curva real'))
    fig.add_trace(go.Scatter(x=xdata, y=ycalc,
                        mode='lines',
                        name='curva calculada'))
    st.write(fig)

def func_log_dashboard(ydata, xdata):
    alpha = st.number_input('Alpha')
    gama = st.number_input('Gama')
    ycalc = models.func_logistica(xdata, gama, alpha)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xdata, y=ydata,
                        mode='lines',
                        name='curva real'))
    fig.add_trace(go.Scatter(x=xdata, y=ycalc,
                        mode='lines',
                        name='curva calculada'))
    st.write(fig)
def tuning_dashboard(ydata, xdata):
    st.title('Faça o tuning de seu modelo')
    st.markdown("""
    ### Tuning modelo SIR
    """)
    length = len(ydata)
    a0_lb = st.number_input('Selecione o limite inferior inicial acumulado de casos de covid')
    a0_ub = st.number_input('Selecione o limite superior inicial acumulado de casos de covid')
    i0_lb = st.number_input('Selecione o limite inferior inicial de infectados')
    i0_ub = st.number_input('Selecione o limite superior inicial de infectados')
    s0_lb = st.number_input('Selecione o limite inferior de pessoas saudaveis')
    s0_ub = st.number_input('Selecione o limite superior de pessoas saudaveis')
    alpha_lb = st.number_input('Limite inferior da taxa de recuperação')
    alpha_ub = st.number_input('Limite superior da taxa de recuperação')
    gama_lb = st.number_input('Limite inferior da taxa de transmissão')
    gama_ub = st.number_input('Limite superior da taxa de transmissão')
    space = ([a0_lb, a0_ub], [s0_lb, s0_ub], [i0_lb, i0_ub], [gama_lb, gama_ub], [alpha_lb, alpha_ub])
    button = st.button('Processo de tuning do modelo SIR')
    if button:
        best_values, goal = optimizer.run_sir_optimizer(ydata, space)
        length = len(ydata)
        ycalc = models.sir_model(length,  best_values[0], best_values[1], best_values[2], best_values[3], best_values[4])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xdata, y=ydata,
                            mode='lines',
                            name='curva real'))
        fig.add_trace(go.Scatter(x=xdata, y=ycalc,
                            mode='lines',
                            name='curva calculada'))
        st.write(fig)
        st.write(goal)
        st.write(best_values)

def main():
    xdata, ydata = load_data()
    options = ['Simulação modelo SIR', 'Simulação função logistica', 'tuning de hiperparametros']
    relatorio = st.sidebar.radio('Escolha um relatorio', options)
    if relatorio == 'Simulação modelo SIR':
        sir_model_dashboard(ydata, xdata)
    elif relatorio == 'Simulação função logistica':
        func_log_dashboard(ydata, xdata)
    else:
        tuning_dashboard(ydata, xdata)

if __name__ == "__main__":
    main()
