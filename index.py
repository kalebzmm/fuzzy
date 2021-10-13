# https://colab.research.google.com/drive/1fQCvZ6w-3q5ZPDy-iDRabAnR7vmq8Dqr?usp=sharing

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Criando as variáveis do problema
temperatura = ctrl.Antecedent(np.arange(0, 41, 1), 'temperatura')
umidade = ctrl.Antecedent(np.arange(0, 101, 1), 'umidade')
velocidade = ctrl.Consequent(np.arange(0, 61, 1), 'velocidade')

# Criando as funções de pertinência para a temperatura
temperatura['fria'] = fuzz.trapmf(temperatura.universe, [0, 0, 10, 22])
temperatura['amena'] = fuzz.gaussmf(temperatura.universe, 22, 5)
temperatura['quente'] = fuzz.trapmf(temperatura.universe, [25, 30, 40, 40])

# Criando as funções de pertinência para a umidade
umidade['baixa'] = fuzz.trapmf(umidade.universe, [0, 0, 20, 60])
umidade['média'] = fuzz.trapmf(umidade.universe, [0, 50, 60, 100])
umidade['alta'] = fuzz.trimf(umidade.universe, [60, 100, 100])

# Criando as funções de pertinência para velocidade
velocidade['lenta'] = fuzz.trapmf(velocidade.universe, [0, 0, 5, 30])
velocidade['normal'] = fuzz.trimf(velocidade.universe, [10, 30, 50])
velocidade['rapida'] = fuzz.trapmf(velocidade.universe, [30, 50, 60, 60])

# Base de Conhecimento/Regras
rule1 = ctrl.Rule(temperatura['fria'] & umidade['baixa'], velocidade['rapida'])
rule2 = ctrl.Rule(temperatura['quente'] | umidade['alta'], velocidade['lenta'])
rule3 = ctrl.Rule(umidade['média'], velocidade['normal'])

# Sistema Fuzzy e Simulação
velocidade_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
velocidade_simulador = ctrl.ControlSystemSimulation(velocidade_ctrl)

# Entrada da temperatura
temp = float(input('Digite a temperatura (°C): '))
velocidade_simulador.input['temperatura'] = temp

# Entrada da umidade
ur = float(input('Digite a umidade (%): '))
velocidade_simulador.input['umidade'] = ur

# Computando o resultado (Inferência Fuzzy + Defuzzificação)
velocidade_simulador.compute()
print('A velocidade é de %d dias' % round(velocidade_simulador.output['velocidade']))

# Visualizando as regiões
temperatura.view(sim=velocidade_simulador)
umidade.view(sim=velocidade_simulador)
velocidade.view(sim=velocidade_simulador)

plt.show()