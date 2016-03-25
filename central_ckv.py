import numpy.matlib as ml
import numpy.linalg as la
import numpy as np
def central_ckv (P,R,c,k,weights,Nd,index,xtest):
    
    one=np.ones(1)
    one=np.expand_dims(one,axis=1)

    P=np.transpose(P)
    ni=index.shape[0]
    index=(index-1).astype(int)    
    
    # normalization of model parameters 
    v_aux=np.diag(1/np.mean(P,axis=1))
    
    # convert model parameters to normalised space
    Normalized_P=np.dot(v_aux,P)
    
    NSupportPoints=Normalized_P.shape[1];  
    Weight_Matrix=np.diag(weights/np.std(Normalized_P,axis=1,ddof=1)); 
    mean_R=np.mean(R,axis=0); 
    mean_R=np.expand_dims(mean_R,axis=1)
    mean_R=np.transpose(mean_R)

    std_R=np.std(R,axis=0,ddof=1);
    std_R=np.expand_dims(std_R,axis=1)

    temp=ml.repmat(mean_R,NSupportPoints,1)
    numer=R-temp

    S=np.array(1/std_R)
    S=np.diag(np.squeeze(S),k=0)
    
    Normalized_R=np.dot(numer,S);

    BasisFxns=np.array([])
    for i in range(0,NSupportPoints): 
        
        for j in range(0,ni):
            ii=index[j]
            temp=np.array([Normalized_P[ii,i]*Normalized_P[index[j:ni],i]])
            temp=np.transpose(temp)

            if (j == 0):
                aux=np.array(temp)
            else:
                aux=np.vstack((aux, temp))
                
        temp=Normalized_P[:,i] 
        temp=np.expand_dims(temp,axis=0)
        aux=np.transpose(aux)
        temp=np.hstack((one,temp, aux))
      
        if (i==0):
            BasisFxns=temp
        else:
            BasisFxns=np.vstack((BasisFxns,temp))
                        
    zhat=sur_model(xtest,v_aux,Weight_Matrix,Normalized_P,Nd,c,k,BasisFxns,Normalized_R,mean_R,std_R,NSupportPoints,index)   
    return zhat


def sur_model(x,v_aux,Weight_Matrix,Normalized_P,Nd,c,k,BasisFxns,Normalized_R,mean_R,std_R,NSupportPoints,index):
    one=np.ones(1)
    one=np.expand_dims(one,axis=1)
    
    # normalize input vector x
    Normalized_X=np.dot(v_aux,x)
    Normalized_X=np.expand_dims(Normalized_X,axis=1)

    Parameter_Diffs=ml.repmat(Normalized_X,1,NSupportPoints)-Normalized_P
    Parameter_Diffs_Weighted=np.dot(Weight_Matrix,Parameter_Diffs)
    
    temp=np.square(Parameter_Diffs_Weighted,Parameter_Diffs_Weighted)
    temp=np.sum(temp,axis=0)
    temp=np.expand_dims(temp,axis=0)
    Distance=np.sqrt(temp);

    SortedDistanceIndex=np.argsort(Distance); 
    SortedDistance=Distance[0,SortedDistanceIndex]    
    D=SortedDistance[0,Nd.astype(int)-1]
    SelectedStorms=SortedDistanceIndex[0,:Nd.astype(int)]

    f=np.exp(-np.power(1/c,k))
    denom=(1-f)
    numer=np.exp(-(np.power(Distance/D/c,k)))-f
    aux1=numer/denom;

    W=np.diag(aux1[0,SelectedStorms]); 

    Ba=BasisFxns[SelectedStorms,:];
    
    ni=index.shape[0]

    for j in range(0,ni):
        ii=index[j]
        temp=np.array([Normalized_X[ii]*Normalized_X[index[j:ni]]])
        temp=np.squeeze(temp)
        temp=np.expand_dims(temp,axis=1)

        if (j == 0):
            aux=np.array(temp)
        else:
            aux=np.vstack((aux, temp))

    b=np.hstack((one,np.transpose(Normalized_X), np.transpose(aux)))

    # auxiliary matrices
    L=np.dot(np.transpose(Ba),W)
    M=np.dot(L,Ba) 
    
    # auxM are the b'*inv(M)*L coefficients in eqn 24 of Taflanidis 2012
    Mi=la.inv(M)
    temp=np.dot(b,Mi)
    auxM=np.dot(temp,L);

    Fi=Normalized_R[SelectedStorms,:];

    temp=np.dot(auxM,Fi)
    std_R=np.transpose(std_R)

    temp2=temp[0,:]*std_R
    f=temp2+mean_R
    f=np.squeeze(f)
    
    return f

