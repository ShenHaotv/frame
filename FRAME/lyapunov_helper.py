import scipy
import numpy as np
from discreteMarkovChain import markovChain
from scipy.linalg.lapack import get_lapack_funcs

""" Construct contrast matrix"""
def getcontrast(o,b):
    contrast=np.zeros((o-1,o))
    for i in range (0,b-1):
        contrast[i,i]=1
        contrast[i,b-1]=-1
    for i in range (b-1,o-1):
        contrast[i,i+1]=1
        contrast[i,b-1]=-1
    return(contrast)

def transformer(d):
    """create a d-dimensinal orthonormalizing P matrix for L"""
    P=np.zeros((d,d))
    P[:,[0]]=np.ones((d,1))/np.sqrt(d)
    for i in range(1,d):
        P[i,i]=-i/np.sqrt(i*i+i)
        for j in range (i):
            P[j,i]=1/np.sqrt(i*i+i)
    return P

def triang_lyapunov(R,F):  
    """Calling lapack and solve the triangulated lyapunov equation"""
    trsyl = get_lapack_funcs('trsyl', (R, F))
    dtype_string = 'T' 
    y, scale, info = trsyl(R, R, F, tranb=dtype_string)
    y *= scale   
    return (y)

def transformed_singular_lyapnov(Lp,y0,G,P,Transpose):                                           
    """Solve the equation LX+XL^T=G,where Transpose=False,  or L^TX+XL=G, where Transpose=True, in the least square sense, the input here has already been transformed"""   
    d=len(y0)                                                                             
    P1=P[:,[0]]
    P2=P[:,1:d]                                                                                                         
    L12p=Lp[[0],1:d]
    L22p=Lp[1:d,1:d]    
      
    if Transpose==False:
       Gp=G-np.sum(np.multiply(y0@y0.T,G))*y0@y0.T                                                                      
       Gp22=P2.T@Gp@P2
       Gp21=P2.T@Gp@P1 
       F=-Gp22
       X_22p=triang_lyapunov(-L22p,F)                                                                                                                     
       X_21p=np.linalg.solve(L22p,Gp21-X_22p@L12p.T) 
       
    elif Transpose==True:        
         Gp22=P2.T@G@P2
         Gp21=P2.T@G@P1
         X_21p=np.linalg.solve(L22p.T,Gp21)
         F=-Gp22+L12p.T@X_21p.T+X_21p@L12p
         X_22p=triang_lyapunov(-L22p.T,F)
    
    X_p=np.zeros((d,d))                                                                                             
    X_p[[0],1:d]=X_21p.T
    X_p[1:d,[0]]=X_21p
    X_p[1:d,1:d]=X_22p
    
    X=P@X_p@P.T                                                                                                     
       
    if Transpose==False:
       X=X-X[0,0]*np.ones((d,d))
    elif Transpose==True:  
         X=X-(X[0,0]/(y0[[0],[0]]**2))*y0@y0.T
       
    return(X)

def transformed_singular_lyapnov_rank1(Lp,y0,c,P,Transpose):                                           
    """Solve the equation LX+XL^T=c@c.T,where Transpose=False,  or L^TX+XL=c@c.T, where Transpose=True, in the least square sense, the input here has already been transformed"""  
    d=len(y0)                                                                                  
    P1=P[:,[0]]
    P2=P[:,1:d]                                                                                                         
    L12p=Lp[[0],1:d]
    L22p=Lp[1:d,1:d]    
     
    if Transpose==False:
       yp1=y0.T@P1
       yp2=P2.T@y0
       product=(y0.T@c)**2                                                                  
       a=P2.T@c
       b=yp2
       F=-(a@a.T-product*b@b.T)
       Gp21=(c.T@P1)*(P2.T@c)-product*yp1*yp2    
       X_22p=triang_lyapunov(-L22p,F)                                                                                                                           
       X_21p=np.linalg.solve(L22p,Gp21-X_22p@L12p.T) 
 
    elif Transpose==True:        
         Gp22=(P2.T@c)@(c.T@P2)
         Gp21=(c.T@P1)*P2.T@c
         X_21p=np.linalg.solve(L22p.T,Gp21)
         F=-Gp22+L12p.T@X_21p.T+X_21p@L12p
         X_22p=triang_lyapunov(-L22p.T,F)
    
    X_p=np.zeros((d,d))                                                                                             
    X_p[[0],1:d]=X_21p.T
    X_p[1:d,[0]]=X_21p
    X_p[1:d,1:d]=X_22p
      
    X=P@X_p@P.T                                                                                                     
       
    if Transpose==False:
       X=X-X[0,0]*np.ones((d,d))
    elif Transpose==True:  
         X=X-(X[0,0]/(y0[[0],[0]]**2))*y0@y0.T
     
    return(X)

def modified_singular_lyapnov(L,gamma,G,Transpose):                
    """Solve the equation diag{gamma}diag{X}+LX+XL^T=G,where Transpose=False, or the transposed equation diag{gamma}diag{X}+L^TX+XL=G, where Transpose=True, exactly"""
    d=len(gamma)
    mc=markovChain(np.identity(d)-L)
    mc.computePi('linear')
    y=mc.pi.reshape((d,1))                                                      #Get the stationary distribution   
    y=y/np.sum(y)
    if np.min(y)<=0:
       mc.computePi('power')
    y=mc.pi.reshape((d,1))                                                      #Get the stationary distribution   
    y=y/np.sum(y)
    y0=y/np.linalg.norm(y)
    y1=y.tolist()

    s = sorted(range(len(y1)), key=lambda k: (y1[k], k), reverse=True)          #Sort the indexes according to the sorting of y              
    s_inv=np.zeros(d)
    for i in range(0,d):
        s_inv[s[i]] = i
       
    s=np.array(s)
    s=s.astype(int)
    s_inv=np.array(s_inv)
    s_inv=s_inv.astype(int)
    y=y[s]
    y0=y0[s]
    L=L[s,:]
    L=L[:,s]
    G=G[s,:]
    G=G[:,s]
    gamma=gamma[s]                                                             #Rearrange y,y0,L,G and gamma according to the sorting

    P=transformer(d)   
    Lp=P.T@L@P
    L22p=Lp[1:d,1:d]   
    
    if Transpose==False:
       z=np.ones((d,1))
       y=y  
       (R,U)=scipy.linalg.schur(-L22p, output='real')    
    
    elif Transpose==True:
         z=y
         y=np.ones((d,1)) 
         (RT,U)=scipy.linalg.schur(-L22p.T, output='real')
         R=RT.T
  
    Z=z@z.T
    Y=y@y.T
    
    Uextended=np.identity(d)
    Uextended[1:d,1:d]=U
    Pnew=P@Uextended                                                            #Combine the real schur decomposition and computing the new transform matrix
    Lp[1:d,1:d]=-R                                                              #Updating Lp
    Lp[:,[0]]=np.zeros((d,1))
    Lp[[0],1:d]=Lp[[0],1:d]@U
    
    X00=transformed_singular_lyapnov(Lp,y0,G,Pnew,Transpose)

    c=np.zeros((d,1))
    c[0]=1/y[0]
    X01=transformed_singular_lyapnov_rank1(Lp,y0,c,Pnew,Transpose)

    gy=np.sum(np.multiply(Y,G))                                                #Sum of entries of the hadmard product of Y and G
    
    X10=X00-gy*X01+gy*(1/(Y[0,0]*Z[0,0]*gamma[0]))*Z

    Aux=np.zeros((d-1,d-1))                                            
    X1_=np.empty((d-1,d,d))

    for j in range(2,d+1):
        c=np.zeros((d,1))
        c[j-1]=1       
        X0j=transformed_singular_lyapnov_rank1(Lp,y0,c,Pnew,Transpose)          
        X1j=X0j-Y[j-1,j-1]*X01+(Y[j-1,j-1]/(Y[0,0]*Z[0,0]*gamma[0]))*Z                  
        X1_[[j-2],:,:]=X1j                                                                           
        for i in range(1,d):
            Aux[i-1,j-2]=gamma[i]*X1j[i,i]                                      #This corresponds to U_{gamma}^T*H1^(-1)U in the method

    a=np.identity(d-1)+Aux
    b=np.zeros((d-1,1))

    for i in range (1,d):
        b[i-1,0]=gamma[i]*X10[i,i]

    x=np.linalg.solve(a,b) 

    S=np.zeros((d,d))

    for i in range(1,d):
        S+=x[[i-1],[0]]*X1_[i-1,:,:]
 
    so=X10-S                                                                    #The solution matrix in the sorted order

    so=so[s_inv,:]                                   
    so=so[:,s_inv]                                                              #The solution matrix in the unsorted order                        
    
    so=0.5*(so+so.T)
    
    return(so)
