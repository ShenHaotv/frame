import numpy as np
import scipy.sparse as sp
from Lyapunov_helper import modified_singular_lyapnov
from discreteMarkovChain import markovChain

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

"""Construct transition matrix from non-zero elements"""
def gettransition(m,M):
    A=M.copy()
    A.data=m                                                                   #Compute transition matrix
    A=A.toarray()
    return (A)

"""Construct laplacian from non-zero elements"""
def getlaplacian(m,M):
    A=M.copy()
    A.data=m                                                                   #Compute laplacian
    A_rowsum=np.array(A.sum(axis=1)).reshape(-1)
    D=sp.diags(A_rowsum).tocsr()
    L=D-A
    return (L)

"""Get the row index and column index of the non-zero element of M"""
def getindex(M):
    nnzm=len(M.data)
    M_index=np.zeros((nnzm,2))                                
    for i in range(len(M.indptr) - 1):
       for k in range (M.indptr[i],M.indptr[i+1]):            
           M_index[k,0]=i
           M_index[k,1]=M.indices[k]
    return(M_index)

def getcoalesce(L,gamma,S,h):                                                  #Get intermediate results of the coalesce process  
    d=len(gamma)
    o=S.shape[0]
    h0=h[0]
    h1=h[1]
    
    s=sorted(range(o), key=lambda k: (h1[k], k), reverse=True)  
    b=s[0]                                                                     
    con=getcontrast(o,b)
    Sc=con@S@con.T
    
    Time=modified_singular_lyapnov(L,gamma,np.ones((d,d)),False)               #Pairwise coalesce time   
    T_ob=Time[h0[:, np.newaxis],h0]                                            #Pairwise coalesce time of the observed demes
    T_bar=T_ob                                                                 #This corresponds to T_bar in the paper
    for i in range (o):
        T_bar[i,i]=T_bar[i,i]*(1-1/h1[i])
     
    Tc=con@T_bar@con.T                                                         #This corresponds to Cob@T_bar@Cob.T in the paper        
    Tc_inv=np.linalg.inv(Tc)                                                   #This corresponds to (Cob@T_bar@Cob.T)^(-1) in the paper    
    quant1=np.trace(-Tc_inv@Sc)
    (sign,absquant2)=np.linalg.slogdet(-Tc)
    quant2=sign*absquant2
    Ttotal=(o-1)/quant1                                                        #Total coalesce time
    lik=quant1+quant2                                                          #Negative log likelihood
    return(Ttotal,Time,T_bar,Tc_inv,lik)
    
def lik_wrapper(z,M,S,h,p,k):
    d=M.shape[0]
    nnzm=len(z)-2
    o=S.shape[0]
    h0=h[0]
    h1=h[1]
      
    s=sorted(range(o), key=lambda k: (h1[k], k), reverse=True)  
    b=s[0]                                                                     
    con=getcontrast(o,b)
    Sc=con@S@con.T
    
    M_index=getindex(M)
    
    m=np.exp(z[0:nnzm])
    c=np.zeros(2)
    c[0]=np.exp(z[nnzm])
    c[1]=1/(1+np.exp(-z[nnzm+1]/k))
    L=getlaplacian(m,M)  
    M=np.identity(d)-L
    mc=markovChain(M)
    mc.computePi('linear')
    y=mc.pi.reshape(d)                                                         #Get the stationary distribution   
    if np.min(y)<=0:
       mc.computePi('power') 
    y=mc.pi.reshape(d)      
    y=y/np.sum(y)
    loggamma=np.log(c[0])-c[1]*np.log(y)
    gamma=np.exp(loggamma)
    dgamma=-c[1]*gamma/y
    
    (Ttotal,Time,T_bar,Tc_inv,lik)=getcoalesce(L,gamma,S,h)
    lik=lik*p/2
         
    l_partial_Tc=Tc_inv@Sc@Tc_inv+Tc_inv  
    l_partial_Tc=l_partial_Tc*p/2
    l_partial_T_bar=con.T@l_partial_Tc@con 
    for i in range (o):
        l_partial_T_bar[i,i]=l_partial_T_bar[i,i]*(1-1/h1[i])
    
    l_partial_T=np.zeros((d,d))                                                #This corresponds topartial l over partial T in the paper
    l_partial_T[h0[:, np.newaxis],h0]=l_partial_T_bar
    
    Time_grad=modified_singular_lyapnov(L,gamma,l_partial_T,True)              #This corresponds to X_grad in the paper
   
    x=np.zeros((d,1))
    for i in range(d):
        x[i]=(-dgamma[i])*Time_grad[i,i]*Time[i,i]
    
    beta=np.linalg.solve(L+np.ones((d,d)),x)
   
    grad_lik_m=np.zeros(nnzm)        
    grad_lik_c=np.zeros(2)
    
    if np.min(Time)>0:                             
       for i in range(nnzm):                                                      #Compute the gradient of the negative log likelihood function with respect to m
           s=int(M_index[i,0])
           t=int(M_index[i,1])            
           Tgs=Time_grad[:,[s]]
           Tt=Time[:,[t]]
           Ts=Time[:,[s]]
           grad_lik_m[i]=2*Tgs.T@(Tt-Ts)+y[s]*(beta[t]-beta[s])

       for i in range(d):                                                         #Compute the gradient of the negative log likelihood function with respect to c     
           if Time_grad[i,i]!=0:
              x=np.log(np.abs(Time_grad[i,i]))+np.log(Time[i,i])
              x_c0=x-c[1]*np.log(y[i])
              grad_lik_c[0]+=-np.sign(Time_grad[i,i])*np.exp(x_c0)
              grad_lik_c[1]+=-np.sign(Time_grad[i,i])*np.exp(x)*(-np.log(y[i])*gamma[i])
    return (lik,grad_lik_m,grad_lik_c)

def pen_wrapper(z,M,deg,lamb):      
    nnzm=len(z)-2
    x=z[0:nnzm]
    m=np.exp(x)  
    m_geomean=np.exp(np.mean(x))                                                #The geometric mean of migration rates   
                                                                                                                
    M_index=getindex(M)
    
    g=m/m_geomean
    dg=1                                                                       #The derivative of g with respect to itself
    
    h=m_geomean/m
    dh=-(m_geomean/m)**2                                                       #The derivative of h with respect to g  
      
    G=gettransition(g,M)                                              
    Gsquare=G**2

    a_g=np.sum(Gsquare+Gsquare.T,axis=1)
    b_g=np.sum(G+G.T,axis=1)
    pen_g=a_g@(1/deg)-(b_g/deg)@(b_g/deg)
      
    H=gettransition(h,M)                                                       
    Hsquare=H**2      

    a_h=np.sum(Hsquare+Hsquare.T,axis=1)
    b_h=np.sum(H+H.T,axis=1)
    pen_h=a_h@(1/deg)-(b_h/deg)@(b_h/deg)
    
    grad_pen_gg=np.zeros(nnzm)
    grad_pen_hh=np.zeros(nnzm)
     
    for i in range(nnzm): 
        s=int(M_index[i,0])
        t=int(M_index[i,1]) 
        grad_pen_gg[i]=2*((1/deg[s]+1/deg[t])*G[s,t]-b_g[s]/(deg[s]**2)-b_g[t]/(deg[t]**2))               #Gradient with respect to g
        grad_pen_hh[i]=2*((1/deg[s]+1/deg[t])*H[s,t]-b_h[s]/(deg[s]**2)-b_h[t]/(deg[t]**2))               #Gradient with respect to h    

    coeff_gm=np.sum(m*grad_pen_gg*dg)/nnzm                                     #Coefficients
    grad_pen_gm=(grad_pen_gg*dg-coeff_gm/m)/m_geomean                          #Gradient with respect to m (first term)      

    coeff_hm=np.sum(m*grad_pen_hh*dh)/nnzm                                     #Coefficients
    grad_pen_hm=(grad_pen_hh*dh-coeff_hm/m)/m_geomean                          #Gradient with respect to m (second term)   
        
    pen=lamb*(pen_g+pen_h)
    grad_pen=lamb*(grad_pen_gm+grad_pen_hm)
    return (pen,grad_pen)

def loss_wrapper(z,M,S,h,p,lamb,deg,k):    
    o=S.shape[0]
    d=M.shape[0]
    nnzm=len(z)-2
    r=300
    coeff=(o-1)*p/(2*d*r) 
    theta=np.exp(z[0:nnzm+1])
    alpha=1/(1+np.exp(-z[nnzm+1]/k))
    coeff_alpha=alpha*(1-alpha)/k
    theta=np.append(theta,coeff_alpha)

    (lik,grad_lik_m,grad_lik_c)=lik_wrapper(z,M,S,h,p,k)
    (pen,grad_pen)=pen_wrapper(z,M,deg,lamb) 
         
    loss=lik+coeff*pen
    
    grad_loss_m=grad_lik_m+coeff*grad_pen
    grad_loss_c=grad_lik_c

    grad_loss= np.append(grad_loss_m,grad_loss_c)                               #Merge the gradient
    grad_loss_z=grad_loss*theta                                                #Gradient with respect to z

    #print(loss)
    return (loss,grad_loss_z)
