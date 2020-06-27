import math
import numpy as np

def sir_model(length, a0, s0, i0, gama, alpha):
  M=210147125
  ak = a0
  sk = s0
  ik = i0
  acumulate = [a0]
  for i in range(1,length):
    skplus1 = sk - (gama * (sk / M) * ik)
    ikplus1 = ik + (gama * (sk / M) * ik) - (alpha * ik)
    akplus1 = ak + (gama * (sk / M) * ik)
    ak = akplus1
    sk = skplus1
    ik = ikplus1
    acumulate.append(ak)

  return np.array(acumulate)

def func_logistica(x_data, alpha, gama):
    M = 210147125
    lista = []
    for x in x_data:
        valor = alpha* M / (1 + (alpha * M - 1)* math.exp(-1*(gama*alpha)* x))
        lista.append(int(valor))
    return np.array(lista)