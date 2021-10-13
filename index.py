# https://colab.research.google.com/drive/1fQCvZ6w-3q5ZPDy-iDRabAnR7vmq8Dqr?usp=sharing

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Criando as variáveis do problema
temperatura = ctrl.Antecedent(np.arange(0, 101, 1), 'temperatura')
umidade = ctrl.Antecedent(np.arange(0, 101, 1), 'umidade')
tempo = ctrl.Consequent(np.arange(0, 73, 1), 'tempo')

# Criando as funções de pertinência para a temperatura
temperatura['fria'] = fuzz.trapmf(temperatura.universe, [0, 0, 10, 22])
temperatura['amena'] = fuzz.gaussmf(temperatura.universe, 22, 5)
temperatura['quente'] = fuzz.trapmf(temperatura.universe, [25, 30, 40, 40])

# Criando as funções de pertinência para a umidade
umidade['baixa'] = fuzz.trapmf(umidade.universe, [0, 0, 20, 60])
umidade['média'] = fuzz.trapmf(umidade.universe, [0, 50, 60, 100])
umidade['alta'] = fuzz.trimf(umidade.universe, [60, 100, 100])

# Criando as funções de pertinência para tempo
tempo['lento'] = fuzz.trapmf(tempo.universe, [30, 50, 60, 72])
tempo['normal'] = fuzz.trimf(tempo.universe, [10, 30, 50])
tempo['rapido'] = fuzz.trapmf(tempo.universe, [0, 0, 5, 30])

# Base de Conhecimento/Regras
rule1 = ctrl.Rule(umidade['alta'] | (umidade['baixa'] & temperatura['quente']), tempo['lento'])
rule2 = ctrl.Rule(temperatura['quente'] | umidade['alta'], tempo['lento'])
rule3 = ctrl.Rule(umidade['média'] & temperatura['amena'], tempo['normal'])

# Sistema Fuzzy e Simulação
tempo_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tempo_simulador = ctrl.ControlSystemSimulation(tempo_ctrl)

# Entrada da temperatura
temp = float(input('Digite a temperatura (°C): '))
tempo_simulador.input['temperatura'] = temp

# Entrada da umidade
ur = float(input('Digite a umidade (%): '))
tempo_simulador.input['umidade'] = ur

# Computando o resultado (Inferência Fuzzy + Defuzzificação)
tempo_simulador.compute()
print('O tempo de germinação é de %d horas' % round(tempo_simulador.output['tempo']))

# Visualizando as regiões
temperatura.view(sim=tempo_simulador)
umidade.view(sim=tempo_simulador)
tempo.view(sim=tempo_simulador)

plt.show()