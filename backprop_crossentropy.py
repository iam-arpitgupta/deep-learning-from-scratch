import numpy as np 
from Backpropagation import SoftmaxActivation
class Loss:
    def cal(self,output,y):
        sample_loss = self.forward(output,y)
        data_loss = np.mean(sample_loss)

        return data_loss
    
# class fot he cross entropy loss 
class CrossEntropyLoss(Loss):
    def forward(self,y_pred, y_true):
        sample = len(y_pred)
        # clip data 
        y_pred_clip = np.clip(y_pred , 1e-2,1e-2)


        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clip[range(sample) , y_true]
            # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
            y_pred_clip*y_true,
            axis=1
            )
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    

    def backward(self , dvalues , y_true):
        # no of the sample is the bascially the number of elem in a row
        samples = len(dvalues)
        # first row of dvalues 
        labels = len(dvalues[0])
        # check the lenght of the y_true if it is == 1 then they are just numbers
        if len(y_true.shape) == 1:
            # convert the number into one hot encoding 
            y_true = np.eye(labels)[y_true]
            # cal gradient -> - ytrue / dvalues 
            self.inputs = - y_true / dvalues

            # normalize the gradients 
            self.dinputs = self.dinputs / samples



# softmax activation backward pass 
class Softmax_crossEntropy:
    def __init__(self):
        self.activation = SoftmaxActivation()
        self.loss = CrossEntropyLoss()


    # forward pass 
    def forward(self , inputs , y_true):
        self.activation.forward()
        self.output = self.activation.output
        return self.loss.cal(self.output ,y)
    
    # backlward pass 
    # for this we need not use the ohe as we need discerete values for this
    # if we have a true class then subract the predited class number to 1 
    # for eg if we have a true class (1,1) that means 1 index of the batch is 1 
    # then subract the 1 form the 1 index of that batch  
    def backward(self,dvalues , y_true):
        sample = len(dvalues)
        # these are the disceret values 
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true ,axis =1)

        self.dinputs = dvalues.copy()
        # range[samples] -> no of total batches or row in the matrix 
        self.dinputs = [range(sample) , y_true] -= 1
        # noramilze gradients
        self.dinputs = self.dinputs / range(sample)

# testing code 
softmax_ouput = np.array([[0.5,0.7,0.8],
                          [0.4,0.8,0.8],
                          [0.5,0.9,0.8]])
class_target = np.array([0,1,1])
soft = Softmax_crossEntropy()
soft.backward(softmax_ouput , class_target)
dvalues = soft.dinputs













