import numpy as np
from sklearn.model_selection import train_test_split


def preprocess_cube(data: np.ndarray) -> np.ndarray:
    #print(data.shape)
    max_depth = 32
    
    new_data = []
    new_data.append(np.swapaxes(data,1,1)[:,:max_depth,:,:])
    new_data.append(np.swapaxes(data,1,2)[:,:max_depth,:,:])
    new_data.append(np.swapaxes(data,1,3)[:,:max_depth,:,:])
                            
    new_data.append(np.flip(np.swapaxes(data,1,1), axis=1)[:,:max_depth,:,:])
    new_data.append(np.flip(np.swapaxes(data,1,2), axis=1)[:,:max_depth,:,:])
    new_data.append(np.flip(np.swapaxes(data,1,3), axis=1)[:,:max_depth,:,:])
            
    return np.array(new_data)


def mergeCube(data):
    modified_data = data.copy()
    for j in range(len(data)):
        k = np.random.randint(0,4)
        modified_data[j] = np.rot90(data[j], k=k, axes=(2,3))
                
    X = np.array(modified_data)[:,:,:1]
    y = np.array(modified_data)[:,:1]
    
    return X, y
    

def split_data(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['Y'], test_size=0.33, random_state=42)
    
    return dict(
        train_x = X_train,
        train_y = y_train,
        test_x = X_test,
        test_y = y_test,
    )
    
    
    
    
    