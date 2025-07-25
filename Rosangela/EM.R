##################################################################
###                       Distribuicao t-student	             ###
###                          Algoritmo EM                      ###
###                         Rosangela Assumpcao                ###
### Criado em: 05/2012                                         ###     
### Atualizado: 04/03/2016                                     ###
##################################################################
ph3=function(k){
   			if (k==0.5){
				phi3<-a/3
				}
			else if (k==1.5){
        			phi3<-a/4.75
					}
			else if (k==2.5){
        		   	phi3<-a/5.92
						}
}
phi3=ph3(k)
	
	c=covar
	theta=c(beta,phi1,phi2,phi3)
	I=diag(1,n,n)
	H <- as.matrix(dist(gr,method="euclidean",diag=TRUE, upper=TRUE))  #Matriz de distancias 
	
EM=function(theta){
t=0
i=1
{
th=matrix(0,10000,(c+5))
while(t==0){
{       
	
## Modelo Fam?lia Mat?rn
 dK=function (H, phi3, k) 
			{
   				if (is.vector(H)) 
        			names(H) <- NULL
    				if (is.matrix(H)) 
        			dimnames(H) <- list(NULL, NULL)
    				uphi <- H/phi3
					uphi <- ifelse(H > 0, -1/2*(besselK(x = uphi, nu = k+1) + besselK(x = uphi, nu = k-1)), 0)
    				return(uphi)
			}

		
Rf3 = matern(H,phi3,k)
Sigma=phi1*I + phi2*Rf3
					
d_phi1= I                      #d R(phi3)/d (phi1)
d_phi2= Rf3     #Fun??o R(phi3)		

M=k*d_phi2 + (1/((2^(k-1))*gamma(k)))*((((H/phi3)^(k+1))*(dK(H,phi3,k))))
d_phi3=phi2*(-(1/phi3)*M)	   #d R(phi3)/d (phi3)

beta.= ginv(t(X)%*%ginv(Sigma)%*%X)%*%t(X)%*%ginv(Sigma)%*%Y  #Estimativa de beta

r = (Y-X%*%beta.)
u=crossprod(forwardsolve(t(chol(Sigma)),r))    #Dist?ncia de mahalanobis
v =((gl+n)/(gl+u))          

	S1=v*(t(r)%*%ginv(Sigma)%*%d_phi1%*%ginv(Sigma)%*%r)
	S2=v*(t(r)%*%ginv(Sigma)%*%d_phi2%*%ginv(Sigma)%*%r)
	S3=v*(t(r)%*%ginv(Sigma)%*%d_phi3%*%ginv(Sigma)%*%r)
    	S=cbind(S1,S2,S3)

	a11=sum(diag(ginv(Sigma)%*%d_phi1%*%ginv(Sigma)))
	a12=sum(diag(ginv(Sigma)%*%d_phi2%*%ginv(Sigma)))
	a13=sum(diag(ginv(Sigma)%*%d_phi3%*%ginv(Sigma)))
	   
 	a21=sum(diag(ginv(Sigma)%*%d_phi1%*%ginv(Sigma)%*%Rf3))
	a22=sum(diag(ginv(Sigma)%*%d_phi2%*%ginv(Sigma)%*%Rf3))

	a33=sum(diag(ginv(Sigma)%*%M%*%ginv(Sigma)%*%(phi2*Rf3)))
	
	A=matrix(c(a11,a21,0,a12,a22,0,a13,0,a33),3,3)
	Fi=S%*%solve(A)
	phi11=Fi[1,1]
	phi22=Fi[1,2]
	Tau=Fi[1,3]
	phi33=-phi22/Tau

theta1=t(cbind(t(beta.),phi11,phi22,phi33))
th[i,1:(c+4)]<-t(theta1)
erro=(sqrt(sum((theta-theta1)^2)))/(sqrt(sum(theta^2)))
th[i,c+5]<-erro

theta=theta1
beta=beta.
phi1=phi11
phi2=phi22
phi3=phi33
i=i+1

	if (erro < e){
					t<-t+1
				   }
}
	#d_phi1 = I          	#d R(phi3)/d (phi1)
	#Rf33=matern(H,phi3,k)
	#d_phi2 = Rf33				        #Fun??o R(phi3)		
	#C = phi1*I+ phi2*Rf33
	#QV <- 
      #MVLV = QV[1,1]
}
print("resultado das itera??es: beta, phi1, phi2, phi3, erro")
IT=print(th[1:i-1,])

#print("Logaritmo da Esperan?a da M?xima Verossimilhan?a")
#print(MVLV) 

}
}

