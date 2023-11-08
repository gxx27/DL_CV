import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 


def generate_batch(batch_size, features, labels):
    """generate batch data function"""

    num_examples = len(labels) # dataset length
    idx = list(range(num_examples))
    for i in range(0,num_examples, batch_size):
        batch_idx = np.array(idx[1:min(i+batch_size, num_examples)])
        yield features[batch_idx], labels[batch_idx]

if __name__ == "__main__":
    x, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # hyperparameters
    BATCH_SIZE = 16
    LR = 1e-3
    EPOCH = 100
    INPUT_DIM = 2
    HIDDEN_DIM = 32
    OUTPUT_DIM = 2
    activate = 'sigmoid'

    # model
    w_1 = np.random.normal(0, 0.01, size=(INPUT_DIM, HIDDEN_DIM))
    b_1 = np.zeros(HIDDEN_DIM)

    w_2 = np.random.normal(0, 0.01, size=(HIDDEN_DIM, OUTPUT_DIM))
    b_2 = np.zeros(OUTPUT_DIM)


    # training
    for _ in range(EPOCH):
        for x, y in generate_batch(BATCH_SIZE, x_train, y_train): # training
            # forward 
            z_1 = np.matmul(x, w_1) + b_1
            
            if activate == 'relu': # relu activate function
                a_1 = np.maximum(0, z_1) 
            elif activate == 'tanh': # tanh activate function
                a_1 = np.tanh(z_1) 
            elif activate == 'sigmoid': # sigmoid activate function
                a_1 = 1 / (1 + np.exp(-z_1)) 
                
            z_2 = np.matmul(a_1, w_2) + b_2
            exp_scores = np.exp(z_2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # backward
            delta_2 = probs
            delta_2[range(len(y)), y] -= 1

            dw_2 = np.matmul(a_1.T, delta_2)
            db_2 = np.sum(delta_2, axis=0)
            
            if activate == 'relu':
                delta_1 = np.matmul(delta_2, w_2.T) * (a_1 > 0)
            elif activate == 'tanh':
                delta_1 = np.matmul(delta_2, w_2.T) * (1 - np.power(a_1, 2))
            elif activate == 'sigmoid':
                delta_1 = np.matmul(delta_2, w_2.T) * (a_1 * (1 - a_1))

            dw_1 = np.dot(x.T, delta_1)
            db_1 = np.sum(delta_1, axis=0)

            # parameter updates
            w_1 -= LR * dw_1
            b_1 -= LR * db_1
            w_2 -= LR * dw_2
            b_2 -= LR * db_2
            
    
    # test
    z_1 = np.matmul(x_test, w_1) + b_1
    a_1 = np.maximum(0, z_1)
    z_2 = np.matmul(a_1, w_2) + b_2
    test_preds = np.argmax(z_2, axis=1)
    test_acc = np.mean(test_preds == y_test)
    print('test accuracy is:',test_acc)

    plt.scatter(x_test[:,0], x_test[:,1], c=test_preds)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Test Results')
    plt.savefig('numpy_result.png')