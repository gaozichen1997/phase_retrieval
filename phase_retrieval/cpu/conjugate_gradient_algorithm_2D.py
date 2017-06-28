import sys
signal = int(sys.argv[1])
n = int(sys.argv[2])
xLower = float(sys.argv[3])
xUpper = float(sys.argv[4])
yLower = float(sys.argv[5])
yUpper = float(sys.argv[6])
h = float(sys.argv[7])

class Conjugate_Gradient(object):
    def __init__(self,signal,numPoints,xLower,xUpper,yLower,yUpper,h):
        self.signal = signal
        self.n = numPoints
        self.xLower = xLower
        self.xUpper = xUpper
        self.yLower = yLower
        self.yUpper = yUpper
        self.h = h

    def createSignal(self,x,y):
        if self.signal == 1:
            if x**2 + y**2 <= 1:
                return -(1-x**2) - (1-y**2) + 2
            else:
                return 0.
        elif self.signal == 2:
            if abs(x) <= 1 and abs(y) <= 1:
                return x + y + 2
            else:
                return 0.
        elif self.signal == 3:
            if x**2 + y**2 <=1:
                return (1-x**2)*(1-y**2)
            else:
                return 0.
        elif self.signal == 4:
            from scipy import pi,sin,sqrt
            if x**2 + y**2 <= 1:
                return sin(sqrt((2*pi*x)**2 + (2*pi*y)**2))
            else:
                return 0.
    def createSupport(self,x,y):
        if self.signal == 1:
            if x**2 + y**2 <= 1:
                return 1
            else:
                return 0
        elif self.signal == 2:
            if abs(x) <=1 and abs(y) <= 1:
                return 1
            else:
                return 0
        elif self.signal == 3:
            if x**2 + y**2 <= 1:
                return 1
            else:
                return 0
        elif self.signal == 4:
            if x**2 + y**2 <= 1:
                return 1
            else:
                return 0

    def run(self):
        import os
        import scipy as sp
        from scipy import fftpack
        from io import BytesIO
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib as mpl
        mpl.rcParams['xtick.labelsize'] = 6
        mpl.rcParams['ytick.labelsize'] = 6
        from matplotlib import cm
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        
        k = 1
        KMAX = 5000
        x = sp.linspace(self.xLower,self.xUpper,self.n,endpoint=True)
        y = sp.linspace(self.yLower,self.yUpper,self.n,endpoint=True)
        X,Y = sp.meshgrid(x,y)
        s0 = sp.array([self.createSignal(x,y) for x,y in zip(sp.ravel(X),sp.ravel(Y))]).reshape(X.shape)
        support = sp.array([self.createSupport(x,y) for x,y in zip(sp.ravel(X),sp.ravel(Y))]).reshape(X.shape)

        fig = plt.figure()
        ax = fig.add_subplot(221, projection='3d')
        ax.set_zlim(s0.min(),s0.max())
        ax.plot_surface(X, Y, s0,rstride=1, cstride=1, cmap=cm.Accent_r, linewidth=0, antialiased=False)
        plt.title("Original signal")

        mag = abs(fftpack.fftn(s0))  # |F(u)|
        iphase = sp.random.uniform(-sp.pi,sp.pi,size=mag.size//2 - 1) #ϕ_k
        iphase = sp.concatenate((sp.array([0]),iphase,sp.array([0]),-iphase[::-1]),axis=0).reshape(mag.shape) 

        Gp_k = mag*sp.exp(1j * iphase)                      #step 2
        gp_k = fftpack.ifftn(Gp_k).real                     #step 3
        g_k = gp_k * support                                #step 4
        
        G_k = fftpack.fftn(g_k)                             #step 1
        Gp_k = mag*sp.exp(1j * sp.angle(G_k))               #step 2
        B = {}
        B['k-1'] = X.size**-2 * (sp.absolute(G_k - Gp_k)**2).sum()   #equation 11
        gp_k = fftpack.ifftn(Gp_k).real                     #step 3
        D = gp_k - g_k  #D_1                                #equation 38
        gpp_k = g_k + self.h*D                                   #equation 36
        g_k = gpp_k * support                               #step 4   (equation 35)
        
        ax = fig.add_subplot(222,projection='3d')
        ax.set_zlim(g_k.min(),g_k.max())
        ax.plot_surface(X,Y,g_k,rstride=1,cstride=1,cmap=cm.Accent_r,linewidth=0,antialiased=False)
        plt.title("1 iteration")

        while k < KMAX:
            G_k = fftpack.fftn(g_k)                           #step 1
            Gp_k = mag * sp.exp(1j *sp.angle(G_k))            #step 2
            B['k'] = X.size**-2 * (sp.absolute(G_k - Gp_k)**2).sum()   #equation 11
            gp_k = fftpack.ifftn(Gp_k).real                   #step 3
            D = gp_k - g_k + (B['k']/B['k-1'])*D              #equation 38
            gpp_k = g_k + self.h*D                                 #equation 36
            g_k = gpp_k * support                             #step 4 (equatioin 35)
            B['k-1'] = B['k']
   
            if X.size**2*B['k'] < .0000001:                    #B_k is the error in fourier domain
                break
            k = k + 1
            if k == 10:
                ax = fig.add_subplot(223,projection='3d')
                ax.set_zlim(g_k.real.min(),g_k.real.max())
                ax.plot_surface(X,Y,g_k.real,rstride=1,cstride=1,cmap=cm.Accent_r,linewidth=0,antialiased=False)
                plt.title("10 iterations")
        g_k = g_k.real
        #print("done")

        ax2 = fig.add_subplot(224, projection='3d')
        ax2.set_zlim(g_k.min(),g_k.max())
        ax2.plot_surface(X, Y, g_k,rstride=1, cstride=1, cmap=cm.Accent_r, linewidth=0, antialiased=False)
        plt.title("{0} iterations".format(k))
        sfile = BytesIO()
        plt.savefig(sfile,format='svg')
        plt.close()
        return b''.join(sfile.getvalue().split(os.linesep.encode()))
        
c = Conjugate_Gradient(signal,n,xLower,xUpper,yLower,yUpper,h)
print(c.run())