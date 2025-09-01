
import numpy as np
# overview of 2d
y_true_check = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

y_pred_clipped_check = np.array([
    [0.2, 0.7, 0.1],
    [0.8, 0.1, 0.1],
    [0.1, 0.2, 0.7]
])

mul_result = y_true_check * y_pred_clipped_check
B = np.sum(mul_result ,axis = 1)
c = - np.log(B)
print(np.mean(c))


# class 
class Loss:
    def cal(self,output,y):
        sample_loss = self.forward(output,y)
        data_loss = np.mean(sample_loss)

        return data_loss
    
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


