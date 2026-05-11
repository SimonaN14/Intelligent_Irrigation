import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

soil_moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'soil_moisture')
temperature = ctrl.Antecedent(np.arange(0, 46, 1), 'temperature')
air_humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'air_humidity')

water_amount = ctrl.Consequent(np.arange(0, 11, 1), 'water_amount')

# Влажност на почвата
soil_moisture['dry'] = fuzz.trapmf(soil_moisture.universe, [0, 0, 20, 40])
soil_moisture['normal'] = fuzz.trimf(soil_moisture.universe, [30, 50, 70])
soil_moisture['wet'] = fuzz.trapmf(soil_moisture.universe, [60, 80, 100, 100])

# Температура
temperature['low'] = fuzz.trapmf(temperature.universe, [0, 0, 10, 18])
temperature['medium'] = fuzz.trimf(temperature.universe, [15, 23, 31])
temperature['high'] = fuzz.trapmf(temperature.universe, [28, 35, 45, 45])

# Влажност на въздуха
air_humidity['low'] = fuzz.trapmf(air_humidity.universe, [0, 0, 25, 45])
air_humidity['medium'] = fuzz.trimf(air_humidity.universe, [35, 55, 75])
air_humidity['high'] = fuzz.trapmf(air_humidity.universe, [65, 80, 100, 100])

# Количество вода
water_amount['low'] = fuzz.trapmf(water_amount.universe, [0, 0, 2, 4])
water_amount['medium'] = fuzz.trimf(water_amount.universe, [3, 5, 7])
water_amount['high'] = fuzz.trapmf(water_amount.universe, [6, 8, 10, 10])

rule1 = ctrl.Rule(soil_moisture['dry'] & temperature['high'], water_amount['high'])
rule2 = ctrl.Rule(soil_moisture['dry'] & temperature['medium'], water_amount['high'])
rule3 = ctrl.Rule(soil_moisture['dry'] & air_humidity['low'], water_amount['high'])

rule4 = ctrl.Rule(soil_moisture['normal'] & temperature['high'] & air_humidity['low'], water_amount['medium'])
rule5 = ctrl.Rule(soil_moisture['normal'] & temperature['medium'], water_amount['medium'])
rule6 = ctrl.Rule(soil_moisture['normal'] & air_humidity['high'], water_amount['low'])

rule7 = ctrl.Rule(soil_moisture['wet'], water_amount['low'])
rule8 = ctrl.Rule(temperature['low'] & air_humidity['high'], water_amount['low'])
rule9 = ctrl.Rule(soil_moisture['normal'] & temperature['low'], water_amount['low'])

rule10 = ctrl.Rule(soil_moisture['dry'] & temperature['low'], water_amount['medium'])
rule11 = ctrl.Rule(soil_moisture['wet'] & temperature['high'], water_amount['medium'])

irrigation_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6,
    rule7, rule8, rule9, rule10, rule11
])

irrigation_simulation = ctrl.ControlSystemSimulation(irrigation_ctrl)

def evaluate_irrigation(soil, temp, humidity):
    simulation = ctrl.ControlSystemSimulation(irrigation_ctrl)

    simulation.input['soil_moisture'] = soil
    simulation.input['temperature'] = temp
    simulation.input['air_humidity'] = humidity

    simulation.compute()

    soil_moisture.view(sim=simulation)
    temperature.view(sim=simulation)
    air_humidity.view(sim=simulation)
    water_amount.view(sim=simulation)

    plt.show()

    water = simulation.output['water_amount']

    if water < 4:
        label = "ниско количество вода"
    elif water < 7:
        label = "средно количество вода"
    else:
        label = "високо количество вода"

    return water, label


test_cases = [
    (20, 35, 25),
    (50, 24, 55),
    (85, 20, 70),
    (40, 38, 30),
    (65, 32, 35)
]

for soil, temp, humidity in test_cases:
    water, label = evaluate_irrigation(soil, temp, humidity)
    print(f"Почва: {soil}% | Температура: {temp}°C | Влажност: {humidity}% -> {water:.2f} л. ({label})")

print(evaluate_irrigation(25, 32, 30))

soil_moisture.view()
temperature.view()
air_humidity.view()
water_amount.view()

plt.show()
