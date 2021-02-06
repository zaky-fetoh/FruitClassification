import matplotlib.patches as mpatches
import sklearn.neural_network as NN
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import skimage.filters





def Cont ( im ) :
    A = cv2.cvtColor( im , cv2.COLOR_BGR2GRAY ) ;
    r ,c = A.shape
    GLCM = np.zeros( (256  , 256)  ); 
    for i in range( r-1 ) :
        for j in range( c-1 ) : 
            GLCM[ A[i,j] , A[i,j+1] ] += 1 ; 
    GLCM += np.transpose(GLCM) ;
    S = sum( sum( GLCM ) ) ;
    GLCM /= S ; 
    return sum([GLCM[i,j]*(i-j)**2 for i in range(256) for j in range( 256) ] ) ;

def Clr ( im ) :
    x,y,z = im.shape ;
    T = skimage.filters.threshold_otsu( im ) 
    B =  im[:,:,0] ;  B = B[B > T].mean()
    G =  im[:,:,1] ; G = G[G > T].mean()
    R =  im[:,:,2] ;  R = R[R > T].mean()
    """
    R /= 255 ; G /= 255 ; B /= 255 ; 
    m,n = max( [R , G , B ] ), min([R , G , B ]) ;
    
    if m == R :
        return ( G-B )/( m - n )
    elif m == G :
        return 2. + (B-R)/( m-n ) ;
    else :
        return 4. + ( R-G ) / (m -n ) 
    """
    return [ R , G , B ] 

def TrData():
    Fv = list();  lb = list() 
    #  A=>Apple , G=>Guava , L=>Litchi  , P=>Pineapple
    
    fold = [ r"E:\Apple\*" , r"E:\Guava\*" , r"E:\Litchi\*"  , r"E:\Pineapple\*" ]
    foldlb = [  'A'    , 'G' , 'L' , 'P'  ] 
    
    N = 400
    for T in range( len( fold ) ) :
        pth = glob( fold[T]  ) ;
        for cnt, p in enumerate( pth ):
            im = cv2.imread( p )
            Fv.append( getFv(im) )
            lb.append( foldlb[T]  ) ;
            if cnt == N :
                break
            
    return Fv ,lb

def getFv( im ) :
    return [Cont(im)]+Clr(im) ;



Fv,lb = TrData() 

Fetlb = [ 'Cont' , 'Red' , 'Green' , 'Blue' ] ;


def VisMat() :
    
    ind = 0 ; 
    for i in range ( 4 ):
        for j in range( 4 ):
            ind += 1
            plt.subplot( 4 , 4 , ind )
            plt.axis( 'off' ) ;
            if( i == j ) :
                continue ;

            
            plt.title( Fetlb[i] +'  VS  ' + Fetlb[j] ) ;
            XA = [ Fv[k][i] for k in range( len( Fv) ) if lb[k] =='A' ] 
            YA = [ Fv[k][j] for k in range( len( Fv) ) if lb[k] =='A' ]
            ax = plt.plot( XA ,YA , 'r+' , label = 'Apple'  ) ;
            
            XG = [ Fv[k][i] for k in range( len( Fv) ) if lb[k] =='G' ] 
            YG = [ Fv[k][j] for k in range( len( Fv) ) if lb[k] =='G' ] 
            plt.plot( XG ,YG , 'b+' , label = 'Guava'  )

            XL = [ Fv[k][i] for k in range( len( Fv) ) if lb[k] =='L' ] 
            YL = [ Fv[k][j] for k in range( len( Fv) ) if lb[k] =='L' ] 
            plt.plot( XL ,YL , 'g+' , label = 'Litchi'  )

            XP = [Fv[k][i] for k in range( len( Fv) ) if lb[k] =='P' ] 
            YP = [Fv[k][j] for k in range( len( Fv) ) if lb[k] =='P' ]             
            plt.plot( XP ,YP , 'm+' , label = 'Pineapple' ) ;

    r  = mpatches.Patch(color='red', label='Apple')
    b  = mpatches.Patch(color='blue', label='Guava')
    g  = mpatches.Patch(color='green', label='Litchi')
    m  = mpatches.Patch(color='magenta', label='Pineapple')

    plt.legend(handles=[r,b,g,m])
            
    plt.show() 

VisMat() 

clf = NN.MLPClassifier( hidden_layer_sizes = ( 10 , 10 , 10 ) ) ;
clf.fit( Fv  , lb ) ;


def TesData():
    Fv = list();  lb = list() 
    #  A=>Apple , G=>Guava , L=>Litchi  , P=>Pineapple
    
    fold = [ r"E:\ValApple\*" , r"E:\ValGuava\*" , r"E:\ValLitchi\*"  , r"E:\ValPineapple\*" ]
    foldlb = [  'A'    , 'G' , 'L' , 'P'  ] 
    Hit = 0 ; Mis = 0 ; 
    N = 20
    for T in range( len( fold ) ) :
        pth = glob( fold[T]  ) ;
        for cnt, p in enumerate( pth ):
            im = cv2.imread( p )
            if clf.predict( [getFv(im)] ) == foldlb[T] :
                    Hit += 1;
            else:
                    Mis += 1;
            
            if cnt == N :
                break
            
    return Hit/( Mis+Hit ) ;


print ( TesData() );






