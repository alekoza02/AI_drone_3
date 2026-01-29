from random import random
from math import exp

class NeuralNetwork:
    def __init__(self):

        # INPUTS
        # 1) to target X
        # 2) to target Y
        # 3) speed X
        # 4) speed Y
        # 5) cos angle
        # 6) sin angle
        # 7) angular speed

        # OUTPUTS
        # 1) Left Power
        # 2) Left Angle
        # 3) Right Power
        # 4) Right Angle
        self.construct_empty()


    def construct_empty(self):
        self.input_layer_n: int = 7
        self.hidden_layer1_n: int = 9
        self.hidden_layer2_n: int = 9
        self.output_layer_n: int = 4

        self.input_layer_w: list[float] = [[random() * 2 - 1 for i in range(self.hidden_layer1_n)] for j in range(self.input_layer_n)]
        self.input_layer_s: list[float] = [0 for i in range(self.input_layer_n)]

        self.links_visual_intensity1 = []

        self.hidden_layer1_w = [[random() * 2 - 1 for i in range(self.hidden_layer2_n)] for i in range(self.hidden_layer1_n)]
        self.hidden_layer1_b = [random() * 2 - 1 for i in range(self.hidden_layer1_n)]
        self.hidden_layer1_s = [0 for i in range(self.hidden_layer1_n)]
        
        self.links_visual_intensity2 = []

        self.hidden_layer2_w = [[random() * 2 - 1 for i in range(self.output_layer_n)] for i in range(self.hidden_layer2_n)]
        self.hidden_layer2_b = [random() * 2 - 1 for i in range(self.hidden_layer2_n)]
        self.hidden_layer2_s = [0 for i in range(self.hidden_layer2_n)]
        
        self.links_visual_intensity3 = []

        self.output_layer_s = [0 for i in range(self.output_layer_n)]
        self.output_layer_b = [random() * 2 - 1 for i in range(self.output_layer_n)]


    def IA_step(self, inputs):

        # visual reset
        self.links_visual_intensity1 = [[0 for i in range(self.input_layer_n)] for j in range(self.hidden_layer1_n)]
        self.links_visual_intensity2 = [[0 for i in range(self.hidden_layer1_n)] for j in range(self.hidden_layer2_n)]
        self.links_visual_intensity3 = [[0 for i in range(self.hidden_layer2_n)] for j in range(self.output_layer_n)]

        self.input_layer_s: list[float] = [0 for i in range(self.input_layer_n)]
        self.hidden_layer1_s = [0 for i in range(self.hidden_layer1_n)]
        self.hidden_layer2_s = [0 for i in range(self.hidden_layer2_n)]
        self.output_layer_s = [0 for i in range(self.output_layer_n)]

        # data loading
        for i in range(self.input_layer_n):
            self.input_layer_s[i] = inputs[i]

        # first hidden layer
        for i in range(self.hidden_layer1_n):
            for j in range(self.input_layer_n):
                self.links_visual_intensity1[i][j] = self.input_layer_s[j] * self.input_layer_w[j][i]
                self.hidden_layer1_s[i] += self.links_visual_intensity1[i][j]
            self.hidden_layer1_s[i] = self.activation_function(self.hidden_layer1_s[i] + self.hidden_layer1_b[i])

        # second hidden layer
        for i in range(self.hidden_layer2_n):
            for j in range(self.hidden_layer1_n):
                self.links_visual_intensity2[i][j] = self.hidden_layer1_s[j] * self.hidden_layer1_w[j][i]
                self.hidden_layer2_s[i] += self.links_visual_intensity2[i][j]
            self.hidden_layer2_s[i] = self.activation_function(self.hidden_layer2_s[i] + self.hidden_layer2_b[i])
        
        # output layer
        for i in range(self.output_layer_n):
            for j in range(self.hidden_layer2_n):
                self.links_visual_intensity3[i][j] = self.hidden_layer2_s[j] * self.hidden_layer2_w[j][i]
                self.output_layer_s[i] += self.links_visual_intensity3[i][j]
            self.output_layer_s[i] = self.activation_function(self.output_layer_s[i] + self.output_layer_b[i])

        return self.output_layer_s


    def activation_function(self, value, type=0):
    
        # Allow negative values (avoid overflow)
        if type == 0:
            return 2.0 / (1.0 + exp(-max(-60.0, min(60.0, value)))) - 1.0
    
        # Sigmoid
        if type == 1:
            return 1.0 / (1.0 + exp(-value))
    
        # RElu
        if type == 2:
            return 0 if value < 0 else value
    


    def get_NN_visual_info(self, size_x, size_y):
        
        padding = 10
        true_size_x = size_x - padding * 2
        true_size_y = size_y - padding * 2

        subdivision_x = true_size_x / 4
        subdivision_y = true_size_y / max(self.input_layer_n, self.hidden_layer1_n, self.hidden_layer2_n, self.output_layer_n)

        circles = []

        circles_input = [
            {
                'x' : padding + 0 * subdivision_x + subdivision_x / 2,
                'y' : padding + i * subdivision_y + subdivision_y / 2,
                'intensity' : self.input_layer_s[i],
                'radius' : min(subdivision_x, subdivision_y) / 2.2
            } for i in range(self.input_layer_n)
        ]
        circles_hidden1 = [
            {
                'x' : padding + 1 * subdivision_x + subdivision_x / 2,
                'y' : padding + i * subdivision_y + subdivision_y / 2,
                'intensity' : self.hidden_layer1_s[i],
                'radius' : min(subdivision_x, subdivision_y) / 2.2
            } for i in range(self.hidden_layer1_n)
        ]
        circles_hidden2 = [
            {
                'x' : padding + 2 * subdivision_x + subdivision_x / 2,
                'y' : padding + i * subdivision_y + subdivision_y / 2,
                'intensity' : self.hidden_layer2_s[i],
                'radius' : min(subdivision_x, subdivision_y) / 2.2
            } for i in range(self.hidden_layer2_n)
        ]
        circles_output = [
            {
                'x' : padding + 3 * subdivision_x + subdivision_x / 2,
                'y' : padding + i * subdivision_y + subdivision_y / 2,
                'intensity' : self.output_layer_s[i],
                'radius' : min(subdivision_x, subdivision_y) / 2.2
            } for i in range(self.output_layer_n)
        ]

        circles.extend(circles_input)
        circles.extend(circles_hidden1)
        circles.extend(circles_hidden2)
        circles.extend(circles_output)

        links = []

        for i in range(len(self.links_visual_intensity1)):
            min_visual_intensity = min(self.links_visual_intensity1[i])
            self.links_visual_intensity1[i] = [j - min_visual_intensity for j in self.links_visual_intensity1[i]]
            max_visual_intensity = max(self.links_visual_intensity1[i])
            if max_visual_intensity != 0:
                self.links_visual_intensity1[i] = [j / max_visual_intensity for j in self.links_visual_intensity1[i]]

        links_input_hidden1 = [
            {
                'start' : [padding + 1 * subdivision_x + subdivision_x / 2, padding + j * subdivision_y + subdivision_y / 2],
                'end' : [padding + 0 * subdivision_x + subdivision_x / 2, padding + i * subdivision_y + subdivision_y / 2],
                'intensity' : self.links_visual_intensity1[j][i],
            } for j in range(self.hidden_layer1_n) for i in range(self.input_layer_n)
        ]

        for i in range(len(self.links_visual_intensity2)):
            min_visual_intensity = min(self.links_visual_intensity2[i])
            self.links_visual_intensity2[i] = [j - min_visual_intensity for j in self.links_visual_intensity2[i]]
            max_visual_intensity = max(self.links_visual_intensity2[i])
            if max_visual_intensity != 0:
                self.links_visual_intensity2[i] = [j / max_visual_intensity for j in self.links_visual_intensity2[i]]

        links_hidden1_hidden2 = [
            {
                'start' : [padding + 2 * subdivision_x + subdivision_x / 2, padding + j * subdivision_y + subdivision_y / 2],
                'end' : [padding + 1 * subdivision_x + subdivision_x / 2, padding + i * subdivision_y + subdivision_y / 2],
                'intensity' : self.links_visual_intensity2[j][i],
            } for j in range(self.hidden_layer2_n) for i in range(self.hidden_layer1_n)
        ]

        for i in range(len(self.links_visual_intensity3)):
            min_visual_intensity = min(self.links_visual_intensity3[i])
            self.links_visual_intensity3[i] = [j - min_visual_intensity for j in self.links_visual_intensity3[i]]
            max_visual_intensity = max(self.links_visual_intensity3[i])
            if max_visual_intensity != 0:
                self.links_visual_intensity3[i] = [j / max_visual_intensity for j in self.links_visual_intensity3[i]]

        links_hidden2_output = [
            {
                'start' : [padding + 3 * subdivision_x + subdivision_x / 2, padding + j * subdivision_y + subdivision_y / 2],
                'end' : [padding + 2 * subdivision_x + subdivision_x / 2, padding + i * subdivision_y + subdivision_y / 2],
                'intensity' : self.links_visual_intensity3[j][i],
            } for j in range(self.output_layer_n) for i in range(self.hidden_layer2_n)
        ]

        links.extend(links_input_hidden1)
        links.extend(links_hidden1_hidden2)
        links.extend(links_hidden2_output)

        return circles, links