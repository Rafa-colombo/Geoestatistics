##################################################################
###                       Distribuicao t-student	             ###
###                        Roteiro de analise                  ###
###                         Rosangela Assumpcao                ###
### Criado em:   10/05/2012                                    ###     
### Atualizado: 08/07/2016                                    ###
##################################################################

##################################################################
####                 Carregando pacotes                       #### 
##################################################################
# Carregando o pacote geoR
require(geoR)

# Carregando o pacote splancs
require(splancs)
require(sp)

# Carregando o pacote Randon Fields
require(RandomFields)

# Carregando o pacote mvtnorm
require(mvtnorm)

# Carregando o pacote matrixcalc - para tra?o de matriz
require(matrixcalc)

# Carregando o pacote pracma - faz  o produto de hadamard
require(pracma)

# Carregando o pacote maxLik - algoritmo NewtonRaphson
require(maxLik)

# Carregando o pacote para assimetria e curtose
require(e1071)

# Carregando o pacote classInt
require(classInt)

# Carregando o pacote snow
require(snow)

# Carregando o pacote MASS
require(MASS)

# Limpa a workspace
rm(list = ls()) 
##############################################################
###              Inserindo o arquivo de dados              ###
##############################################################
# Lendo o arquivo de dados
Dados<-read.table("E:/1 UTFPR/PESQUISAS/EM_py/dados.txt")
Dados
P <-as.geodata(Dados,coords.col=1:2, data.col=5,head= TRUE,covar.col=3:4)  #,
P
boxplot(P$data)
###########################################################
###             Construcao do semivariograma            ###
###########################################################
max(dist(P$coords))
min(dist(P$coords))
d.var <- variog(P,uvec=seq(1,1200,l=10),estimador.type="classical", pairs.min=30, direction="omnidirectional",tolerance=pi/8)
plot(d.var, xlab="dist?ncia", ylab="semivari?ncia",main=' ',font.main = 3)

###########################################################
###             Estimativa dos parametros         		  ###
###					usando o algoritmo EM                       ###
###########################################################
### FIXANDO OS PAR?METROS UTILIZADOS  
gr=P$coords   # grid
Y=P$data      # variável resposta Y
n=length(Y)   # tamanho do conjunto de dados
X=matrix(rep(1,n),n,1) # matriz das covariáveis X
beta=solve(t(X)%*%X)%*%t(X)%*%Y  # coeficientes do modelo
phi1=0.002                       # phi1, phi2 e phi3 parâmetros do modelo espacial
phi2=0.004
a=400                            # alcance a é determinado em função do modelo espacial
modelo="matern"                  # modelo espacial "matern"
covar=0                          # covariância

gl=3                             # grau de liberdade da distribuição t-student
k=2.5                            # kappa do modelo matern
e=0.00005                        # erro de parada
## Ler o arquivo com algoritmo EM
source("E:/1 UTFPR/PESQUISAS/EM_py/EM.R")                   # EM algoritmo que estima os parâmetros espaciais phi1, phi2 e phi3
est.EM=EM(theta)                 # theta - vetor com os parâmetros espaciais

E=matrix(est.EM[50,],1,5)
##########################################################
#Construcao da linha no semivariograma com os parâmetros estimados
#beta=as.matrix(c(E[,1], E[,2], E[,3]),1,3)
beta=as.matrix(E[,1],1,1)
phi1=E[,2]        
phi2=E[,3]     
phi3=E[,4] 
theta=c(t(beta),phi1,phi2,phi3)
h=c(0:1200)
Rf3s=matern(h,phi3,k)
P.t=phi1+phi2*(1-Rf3s)
par=cbind(h,P.t)
#par

lines(P.t,col='red')    #k=0.5
lines(P.t,col='blue')   #k=1.5
lines(P.t,col='green')  #k=2.5

###########################################################
##                       CV                             ###
###########################################################
## L? o arquivo com Valida??o cruzada
source("CVp.R")
cross.valid


###########################################################
##                    erro padr?o                       ###
###########################################################
source("EP.R")

#####################################################################
##            Influ?ncia local Cook - Zhu - Pan                   ###
#####################################################################
source("pan.R")
#####################################################################
## Gr?fico C1i  -  eq(8) Pan
plot(C1i,xlab="Ordem",ylab=expression(C1i),pch=19,font.main=4)
identify(C1i,offset=0.3, cex=1.2)

####################################################################
## Gr?fico CQi - equa??o (4) (Pan, 2014) trocando L por Q
####################################################################
plot(CQi,xlab="Ordem",ylab=expression(CQi),pch=19,font.main=4, ylim=c(0.02168,0.0221))
identify(CQi,offset=0.3, cex=1.2)

####################################################################
## Gr?fico CQei - Equa??o (5) (Pan, 2015) trocando L por Q
####################################################################
plot(CQei,xlab="Ordem",ylab=expression(CQei),pch=19,font.main=4)
identify(Di,offset=0.3, cex=1.2)

#####################################################################
## Gr?fico Di - eq(9) Pan
#####################################################################
plot(Di,xlab="Ordem",ylab=expression(Di),pch=19,font.main=4)
identify(Di,offset=0.3, cex=1.2)

#####################################################################
## Gr?fico Die - equa??o 10 (Pan, 2014)		
#####################################################################
plot(Die,xlab="Ordem",ylab=expression(Die),pch=19,font.main=4)
identify(Die,offset=0.3, cex=1.2)

###########################################################
##                   Krigagem                           ###
###########################################################
bor=read.table("bordas_jt.txt", head=F)
plot(bor)
polygon(bor) 
points(P$coords)
apply(bor,2,range) #mostra m?nimo e m?ximo das coordenadas, para determinar o grid de interpola??o 
gi <- expand.grid(x=seq(7.615, 28.966, by=0.5), y=seq(5.711,66.411, by=1)) 
plot(gi)
require(splancs)
gs <- polygrid(gi,bor=bor)
points(gs, pch="+", col=2)

dim(gs)
L=1453

source("krig.R")
Z0=Predict[,3]
suppressWarnings(image(Z0, loc=gs, border=bor,col=gray(seq(1,0,l=5))))

