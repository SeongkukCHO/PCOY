#####Source Code No.1


########################################################################################
#                                  Assignment2                                         #
#                          Python Programming Language                                 #
#                     Building Neural Network-based XOR logic                          #
#                             Due to April 17, 2019(Wed)                               #
########################################################################################
#                      Building Neural Network-based XOR logic                         #
#                             Author   : Cho seongkuk                                  #
#                             PL       : Python                                        #
#                             OS       : Linux                                         #
########################################################################################
# 1) 과제 목표 : Neural Network에 대한 이해 및 NNs를 활용한 XOR logic에 대한 학습 결과 도출#
# 2) 조건 : 2 input units * 2 hidden units * 1 ouput unit                              #
########################################################################################
#   문제점 : 2개의 hidden node로는 local minima에 쉽게 빠져 학습이 어려움을 발견           #
#   해결 방안 : Acc값을 통해 적절한 hidden node수를 찾아낸다.                             #
########################################################################################
#                                  Revision History                                    #
########################################################################################  

# ① Adding an Acc[] variable                                                          
#  feature 1) Accuracy is stored(Acc = 1-abs(output error))                            
#  feature 2) 해당 var를 통해 적당한 hidden node의 수를 정한다.                            
#                                                                                      
# ② The result of train() is returned.                                                
#  feature 1) output의 결과를 plot()을 통해 확인하기 위함                                  
#                                                                                      
# ③ 적당한 Hidden node 수를 찾기 위한 검색                                               
#  feature 1) 2개의 hidden node로는 연산이 부족함을 발견하여 점차 증가하여 적당한 node 수 결정 
#  feature 2) 적당한 node의 수는 Acc의 값이 0.95(즉, 95% 이상의 정확도)를 보일 때 정지       

########################################################################################
#                            Import Package                                            #
########################################################################################

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm

np.random.seed(1) # report submission을 위한 seed값을 활용
########################################################################################
#                              Function                                                #
########################################################################################

def sigmoid(x) : 
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
########################################################################################
#                              Variable                                                #
########################################################################################

Acc=[] #Accuracy
Input_Data= np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
Output_Data = np.array([[0],[1], [1], [0]])

########################################################################################
#                          Class Definitions                                           #
########################################################################################

class NeuralNetwork: 
    def __init__(self,  
                 no_of_in_nodes,  
                 no_of_out_nodes,  
                 no_of_hidden_nodes, 
                 learning_rate): 
        self.no_of_in_nodes = no_of_in_nodes 
        self.no_of_out_nodes = no_of_out_nodes 
        self.no_of_hidden_nodes = no_of_hidden_nodes 
        self.learning_rate = learning_rate  
        self.create_weight_matrices() 
         
    def create_weight_matrices(self): 
        """ A method to initialize the weight matrices of the neural network""" 
        rad = 1 / np.sqrt(self.no_of_in_nodes) 
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad) 
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes,  
                                       self.no_of_in_nodes)) 
        rad = 1 / np.sqrt(self.no_of_hidden_nodes) 
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad) 
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes,  
                                        self.no_of_hidden_nodes))    
         
     
    def train(self, input_vector, target_vector): 

        # input_vector and target_vector can be tuple, list or ndarray 
        input_vector = np.array(input_vector, ndmin=2).T 
        target_vector = np.array(target_vector, ndmin=1).T 
         
        #hidden layer1 
        output_vector1 = np.dot(self.weights_in_hidden, input_vector) 
        output_vector_hidden = activation_function(output_vector1) 
        #hidden layer2 
        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden) 
        output_vector_network = activation_function(output_vector2) 
        #errors 
        output_errors = target_vector - output_vector_network
 
        #Added Portion#######
        Acc.append(np.mean(1-(np.abs(output_errors)))) 
        #################
         
        # update the weights: 
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)      
        tmp = self.learning_rate  * np.dot(tmp, output_vector_hidden.T) 
        self.weights_hidden_out += tmp 
        # calculate hidden errors: 
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors) 
        # update the weights: 
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden) 
        self.weights_in_hidden += self.learning_rate * np.dot(tmp, input_vector.T)
        
        return output_vector_network 
          
    def run(self, input_vector): 
        # input_vector can be tuple, list or ndarray 
        input_vector = np.array(input_vector, ndmin=2).T 
        output_vector = np.dot(self.weights_in_hidden, input_vector) 
        output_vector = activation_function(output_vector) 
         
        output_vector = np.dot(self.weights_hidden_out, output_vector) 
        output_vector = activation_function(output_vector) 
     
        return output_vector 


########################################################################################
#                            Main() Function                                           #
########################################################################################

#### first for() : hidden node 수 결정
for count in range(1,1000) :
    simple_network = NeuralNetwork(no_of_in_nodes=2, 
                               no_of_out_nodes=1, 
                               no_of_hidden_nodes=count,
                               learning_rate=0.7)
    x,y=[],[]


    #### second for() : Input data와 Output data를 활용한 50000번의 iteration(50000번 학습)
    for i in range(50000):
        traning_cls=simple_network.train(Input_Data, Output_Data)
        x.append(traning_cls[0])
        y.append(i)
        if(i%10000==0) : print(i,": ",traning_cls)
    
    acc = Acc[49999] #최종적으로 저장된 값이 Acc
    print("\n\nno_of_hidden_nodes : ",count,"\nAccuracy  : ", acc)

    #### first plot() : 훈련량에 따른 Acc 변화량 그래프
    plt.subplot(2,1,1)
    plt.plot(y,Acc)
    plt.title("ACC")
    
    #### second plot() : input data들의 훈련량에 따른 output 변화 그래프
    plt.subplot(2,1,2)
    plt.plot(y,x)
    plt.plot("Result")
    plt.show()
    
    #### hidden node수가 늘어났을 때(즉, 다음 for문이 반복될 때)를 위한 변수 초기화
    Acc.clear(), y.clear(), x.clear()
    
    #### Accuracy가 0.95(즉, 95%이상의 정확도를 보일 때)이상이면 break
    if(acc > 0.95) :        
        break

#### Input Data를 랜덤배치 한 test data를 통한 모델 확인
#### int(round(cls1[0][i], 0)) : 최종 output에서 반올림하여 결과값 확인
np.random.shuffle(Input_Data)
cls1 = simple_network.run(Input_Data)
for i in range(len(Input_Data)) :
    print(Input_Data[i][0]," XOR ",Input_Data[i][1],"는 ",int(round(cls1[0][i], 0)))
    
    
    
###############################################################################################
    
    
#####Source Code No.2
    
    
########################################################################################
#                            Assignment2                                               #
#                       Python Programming Language                                    #
#                Building Neural Network-based XOR logic                               #
#                      Due to April 17, 2019(Wed)                                      #
########################################################################################
#                 Building Neural Network-based XOR logic                              #
#                      Author   : Cho seongkuk                                         #
#                      PL       : Python                                               #
#                      OS       : Linux                                                #
########################################################################################
# 1) 과제 목표 : Neural Network에 대한 이해 및 NNs를 활용한 XOR logic에 대한 학습 결과 도출#
# 2) 조건 : 2 input units * 2 hidden units * 1 ouput unit                              #
########################################################################################
#   문제점 : 2개의 hidden node로는 local minima에 쉽게 빠져 학습이 어려움을 발견           #
#   해결 방안 : 1개의 hidden layer 추가(동일 문제 발생하여 hideen node를 늘림)            #
########################################################################################
#                               Revision History                                       #
########################################################################################
# ① Add to Second hidden layer
#  feature 1) 기존 소스코드와 동일한 방식으로 두 번째 hidden layer를 추가한다.
#  feature 2) 가중치 조절 또한 마찬가지로 기존 소스코드와 동일한 방식을 활용한다. 
#
# ② Adding an Acc[] variable                                                          
#  feature 1) Accuracy is stored(Acc = 1-abs(output error))                            
#  feature 2) 해당 var를 통해 적당한 hidden node의 수를 정한다.                            
#                                                                                      
# ③ The result of train() is returned.                                                
#  feature 1) output의 결과를 plot()을 통해 확인하기 위함                                  
#                                                                                      
# ④ 적당한 Hidden node 수를 찾기 위한 검색                                               
#  feature 1) 2개의 hidden node로는 연산이 부족함을 발견하여 점차 증가하여 적당한 node 수 결정 
#  feature 2) 적당한 node의 수는 Acc의 값이 0.95(즉, 95% 이상의 정확도)를 보일 때 정지       

########################################################################################
#                            Import Package                                            #
########################################################################################

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm

np.random.seed(1) # report submission을 위한 seed값을 활용
########################################################################################
#                              Function                                                #
########################################################################################

def sigmoid(x) : 
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
########################################################################################
#                              Variable                                                #
########################################################################################

Acc=[] #Accuracy
w_h=[], w_h_2=[], w_h_o=[] //weights
Input_Data= np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
Output_Data = np.array([[0],[1], [1], [0]])

########################################################################################
#                          Class Definitions                                           #
########################################################################################

class NeuralNetwork:
    def __init__(self, 
                 no_of_in_nodes, 
                 no_of_out_nodes, 
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate 
        self.create_weight_matrices()
        
    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural network"""
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, 
                                       self.no_of_in_nodes))
        
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden_2 = X.rvs((self.no_of_hidden_nodes, 
                                        self.no_of_hidden_nodes))   
        
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes, 
                                        self.no_of_hidden_nodes))   
        
    
    def train(self, input_vector, target_vector):

        #####  input_vector and target_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=1).T
        
        #####  1
        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_vector_hidden1 = activation_function(output_vector1)
        w_h.append(self.weights_in_hidden)
    
        #####  2
        output_vector2 = np.dot(self.weights_in_hidden_2, output_vector_hidden1)
        output_vector_hidden2 = activation_function(output_vector2)
        w_h_2.append(self.weights_in_hidden_2)
    
        #####  3
        output_vector3 = np.dot(self.weights_hidden_out, output_vector_hidden2)
        output_vector_network = activation_function(output_vector3)
        w_h_o.append(self.weights_hidden_out)
        
        ##### errors
        output_errors = target_vector - output_vector_network
 
        #Added Portion#######
        Acc.append(np.mean(1-(np.abs(output_errors)))) 
        #################
        
        ####  update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)     
        tmp = self.learning_rate  * np.dot(tmp, output_vector_hidden2.T)
        self.weights_hidden_out += tmp
        ####  calculate hidden errors:
        hidden1_errors = np.dot(self.weights_hidden_out.T, output_errors)
        
        ####  update the weights:        
        tmp = hidden1_errors * output_vector_hidden2 * (1.0 - output_vector_hidden2)     
        tmp = self.learning_rate  * np.dot(tmp, output_vector_hidden1.T)
        self.weights_in_hidden_2 += tmp
        ####  calculate hidden errors:
        hidden2_errors = np.dot(self.weights_in_hidden_2.T, hidden1_errors)
        
        #### update the weights:
        tmp = hidden2_errors * output_vector_hidden1 * (1.0 - output_vector_hidden1)
        self.weights_in_hidden += self.learning_rate * np.dot(tmp, input_vector.T)
        
        return output_vector_network
    
    def run(self, input_vector):
        #### input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T
        
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = activation_function(output_vector)
        
        output_vector = np.dot(self.weights_in_hidden_2, output_vector)
        output_vector = activation_function(output_vector)
        
        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = activation_function(output_vector)
        return output_vector

########################################################################################
#                            Main() Function                                           #
########################################################################################

#### first for() : hidden node 수 결정
for count in range(1,1000) :
    simple_network = NeuralNetwork(no_of_in_nodes=2, 
                               no_of_out_nodes=1, 
                               no_of_hidden_nodes=count,
                               learning_rate=0.7)
    x,y=[],[]
    
    #### second for() : Input data와 Output data를 활용한 50000번의 iteration(50000번 학습)
    for i in range(50000):
        traning_cls=simple_network.train(Input_Data, Output_Data)
        x.append(traning_cls[0])
        y.append(i)
        if(i%10000==0) : print(i,": ",traning_cls)
    
    acc = Acc[49999] #최종적으로 저장된 값이 Acc
    print("\n\nno_of_hidden_nodes : ",count,"\nAccuracy  : ", acc)
    print("weight_hidden = ", w_h[49999])
    print("weight_hidden_out = ", w_h_2[49999])
    print("weight_hidden_out = ", w_h_o[49999])

    #### first plot() : 훈련량에 따른 Acc 변화량 그래프
    plt.subplot(2,1,1)
    plt.plot(y,Acc)
    plt.title("ACC")
    
    #### second plot() : input data들의 훈련량에 따른 output 변화 그래프
    plt.subplot(2,1,2)
    plt.plot(y,x)
    plt.plot("Result")
    plt.show()
    
    #### hidden node수가 늘어났을 때(즉, 다음 for문이 반복될 때)를 위한 변수 초기화
    Acc.clear(), y.clear(), x.clear(),w_h.clear(), w_h_2.clear(), w_h_o.clear()
    
    #### Accuracy가 0.95(즉, 95%이상의 정확도를 보일 때)이상이면 break
    if(acc > 0.95) :        
        break

#### Input Data를 랜덤배치 한 test data를 통한 모델 확인
#### int(round(cls1[0][i], 0)) : 최종 output에서 반올림하여 결과값 확인
np.random.shuffle(Input_Data)
cls1 = simple_network.run(Input_Data)
for i in range(len(Input_Data)) :
    print(Input_Data[i][0]," XOR ",Input_Data[i][1],"는 ",int(round(cls1[0][i], 0)))
    
    
    
    
    
    
    
    
################################################################################################3


#####Source Code No.3
    
    
    
########################################################################################
#                            Assignment2                                               #
#                       Python Programming Language                                    #
#                Building Neural Network-based XOR logic                               #
#                      Due to April 17, 2019(Wed)                                      #
########################################################################################
#                 Building Neural Network-based XOR logic                              #
#                        Author   : Cho seongkuk                                       #
#                        PL       : Python                                             #
#                        OS       : Linux                                              #
########################################################################################
# 1) 과제 목표 : Neural Network에 대한 이해 및 NNs를 활용한 XOR logic에 대한 학습 결과 도출#
# 2) 조건 : 2 input units * 2 hidden units * 1 ouput unit                              #
########################################################################################
#   문제점 : local minima문제를 해결한다                                                 #
#   해결 방안 : 적은 learning rate값으로 weight 변화량을 조금씩 변화시킴                   #
########################################################################################
#                                             Import Package                           #
########################################################################################

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm

np.random.seed(201904) # report submission을 위한 seed값을 활용

########################################################################################
#                              Function                                                #
########################################################################################

def sigmoid(x) : 
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
########################################################################################
#                              Variable                                                #
########################################################################################

errors=[]
Input_Data= np.array([[0.99, 0.99], [0.99, 0.01], [0.01, 0.99], [0.01, 0.01]]) 
Output_Data = np.array([[0.01],[0.99], [0.99], [0.01]]) 

########################################################################################
#                          Class Definitions                                           #
########################################################################################

class NeuralNetwork: 
    def __init__(self,  
                 no_of_in_nodes,  
                 no_of_out_nodes,  
                 no_of_hidden_nodes, 
                 learning_rate): 
        self.no_of_in_nodes = no_of_in_nodes 
        self.no_of_out_nodes = no_of_out_nodes 
        self.no_of_hidden_nodes = no_of_hidden_nodes 
        self.learning_rate = learning_rate  
        self.create_weight_matrices() 
         
    def create_weight_matrices(self): 
        """ A method to initialize the weight matrices of the neural network""" 
        rad = 1 / np.sqrt(self.no_of_in_nodes) 
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad) 
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes,  
                                       self.no_of_in_nodes)) 
        rad = 1 / np.sqrt(self.no_of_hidden_nodes) 
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad) 
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes,  
                                        self.no_of_hidden_nodes))    
    def train(self, input_vector, target_vector): 

        # input_vector and target_vector can be tuple, list or ndarray 
        input_vector = np.array(input_vector, ndmin=2).T 
        target_vector = np.array(target_vector, ndmin=1).T 
         
        #hidden layer1 
        output_vector1 = np.dot(self.weights_in_hidden, input_vector) 
        output_vector_hidden = activation_function(output_vector1) 
        #hidden layer2 
        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden) 
        output_vector_network = activation_function(output_vector2) 
        #errors 
        output_errors = target_vector - output_vector_network

        errors.append(np.mean(abs(output_errors)))
         
        # update the weights: 
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)      
        tmp = self.learning_rate  * np.dot(tmp, output_vector_hidden.T) 
        self.weights_hidden_out += tmp 
        # calculate hidden errors: 
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors) 
        # update the weights: 
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden) 
        self.weights_in_hidden += self.learning_rate * np.dot(tmp, input_vector.T) 
          
     
    def run(self, input_vector): 
        # input_vector can be tuple, list or ndarray 
        input_vector = np.array(input_vector, ndmin=2).T 
        output_vector = np.dot(self.weights_in_hidden, input_vector) 
        output_vector = activation_function(output_vector) 
         
        output_vector = np.dot(self.weights_hidden_out, output_vector) 
        output_vector = activation_function(output_vector) 
     
        return output_vector 

########################################################################################
#                            Main() Function                                           #
########################################################################################

simple_network = NeuralNetwork(no_of_in_nodes=2,
                               no_of_out_nodes=1,  
                               no_of_hidden_nodes=2, 
                               learning_rate=0.017) 

i=0
while True :
    i+=1
    traning_cls=simple_network.train(Input_Data, Output_Data)
    if(i%1000==0) : print("error of ",i,"th iteration is ", errors[i-1])
    if(errors[i-1] < 0.05) : break;
    
plt.plot(errors)     

np.random.shuffle(Input_Data)
cls1 = simple_network.run(Input_Data) 
for i in range(len(Input_Data)) : 
    print(int(round(Input_Data[i][0]))," XOR ",int(round(Input_Data[i][1])),"=",int(round(cls1[0][i], 0)))
