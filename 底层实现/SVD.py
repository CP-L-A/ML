import numpy as np
def load_data():
    data=np.random.rand(5,3)
    return np.mat(data)

if __name__=='__main__':
    data=load_data()
    U,Sigma,D=np.linalg.svd(data)
    print(U)
    print(Sigma)
    print(D)