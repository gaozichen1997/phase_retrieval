import sys
try:
    signal = int(sys.argv[1])
    n = int(sys.argv[2])
    xLower = float(sys.argv[3])
    xUpper = float(sys.argv[4])
    yLower = float(sys.argv[5])
    yUpper = float(sys.argv[6])
    h = float(sys.argv[7])
except:
    signal = 1
    n = 8
    xLower = -1.6
    xUpper = 1.6
    yLower = -1.6
    yUpper = 1.6
    h = 1.1

class Conjugate_Gradient(object):
    def __init__(self,signal,numPoints,xLower,xUpper,yLower,yUpper,h):
        self.signal = signal
        self.n = numPoints
        self.xLower = xLower
        self.xUpper = xUpper
        self.yLower = yLower
        self.yUpper = yUpper
        self.h = h
    
    ######################### Set up variables ######################################    
    #os.environ["PYOPENCL_COMPILER_OUTPUT"] = str(0)
    ####################################################################################
    
    def createSignal(self,x,y):
        from scipy import sin,sqrt,pi
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
        from reikna import cluda
        from reikna.cluda import functions, dtypes
        from reikna.fft import FFT
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib as mpl
        mpl.rcParams['xtick.labelsize'] = 6
        mpl.rcParams['ytick.labelsize'] = 6
        from matplotlib import cm
        import matplotlib.pyplot as plt
        from io import BytesIO
        
        api = cluda.any_api()  
        thr = api.Thread.create()
        k = 1
        KMAX = 5000  #number of total iterations
        dtype = [sp.array([1.2]).dtype,sp.array([1j]).dtype]
        
        x = sp.linspace(self.xLower,self.xUpper,self.n,endpoint=True)
        y = sp.linspace(self.yLower,self.yUpper,self.n,endpoint=True)
        X,Y = sp.meshgrid(x,y)
        s0 = sp.array([self.createSignal(x,y) for x,y in zip(sp.ravel(X),sp.ravel(Y))]).reshape(X.shape)
        support = sp.array([self.createSupport(x,y) for x,y in zip(sp.ravel(X),sp.ravel(Y))]).reshape(X.shape)
        
        fig = plt.figure()
        ax = fig.add_subplot(221, projection='3d')
        ax.set_zlim(s0.min(),s0.max())
        ax.plot_surface(X, Y, s0,rstride=1, cstride=1, cmap=cm.Accent_r,
                               linewidth=0, antialiased=False)
        plt.title("Original signal")
        
        mag = abs(fftpack.fftn(s0))  # |F(u)|
        
        iphase = sp.random.uniform(-sp.pi,sp.pi,size=mag.size//2 - 1) #ϕ_k
        iphase = sp.concatenate((sp.array([0]),iphase,sp.array([0]),-iphase[::-1]),axis=0)
        iphase = iphase.reshape(mag.shape)
        
        
        WORK_SIZE = mag.size if mag.size <= 2**8 else 2**8
        
        
        Gp_k = mag*sp.exp(1j * iphase)                      #step 2
        gp_k = fftpack.ifftn(Gp_k).real                     #step 3
        g_k = gp_k * support                                #step 4
        G_k = fftpack.fftn(g_k)                             #step 1
        Gp_k = mag*sp.exp(1j * sp.angle(G_k))               #step 2
        Bk1,Bk = 0,0                                        # [B_{k-1}, B_k]
        Bk1 = X.size**-2 * (sp.absolute(G_k - Gp_k)**2).sum()   #equation 11

        gp_k = fftpack.ifftn(Gp_k).real                     #step 3
        D = gp_k - g_k  #D_1                                #equation 38
        gpp_k = g_k + self.h*D                                   #equation 36
        g_k = gpp_k * support                               #step 4   (equation 35)
        
        ax = fig.add_subplot(222,projection='3d')
        ax.set_zlim(g_k.min(),g_k.max())
        ax.plot_surface(X,Y,g_k,rstride=1,cstride=1,cmap=cm.Accent_r,linewidth=0,antialiased=False)
        plt.title("1 iteration")
        
        mag_dev = thr.to_device(mag.astype(dtype[0]))
        Gp_k_dev = thr.to_device(Gp_k.astype(dtype[1]))
        gp_k_dev = thr.to_device(gp_k.astype(dtype[0]))
        g_k_dev = thr.to_device(g_k.astype(dtype[1]))
        G_k_dev = thr.to_device(G_k.astype(dtype[1]))
        gpp_k_dev = thr.to_device(gpp_k.astype(dtype[0]))
        D_dev = thr.to_device(D.astype(dtype[0]))
        B_dev = thr.to_device(sp.array([Bk1,Bk]).astype(dtype[0]))  # [B_{k-1}, B_k]
        tempArray_dev = thr.empty_like(mag.astype(dtype[0]))
        support_dev = thr.to_device(support.astype(dtype[0]))
        
        
        cfft = FFT(G_k_dev).compile(thr)
        prg1 = thr.compile("""
            KERNEL void makeGp_k(
                GLOBAL_MEM double *mag,
                GLOBAL_MEM ${ctype} *G_k,
                GLOBAL_MEM ${ctype} *Gp_k)
            {
                const SIZE_T i = get_local_id(0);
                const SIZE_T group = get_group_id(0);
                const SIZE_T workSize = get_local_size(0);
                const SIZE_T index = group*workSize + i;
                double arg = atan2(G_k[index].y, G_k[index].x);   // the argument of a complex number
                Gp_k[index] = ${polar}(mag[index], arg);          // r*e^(i*theta)
            }
    
            KERNEL void makeB_kHelper(
                GLOBAL_MEM ${ctype} *G_k,
                GLOBAL_MEM ${ctype} *Gp_k,
                GLOBAL_MEM double *tempArray)
            {
                const SIZE_T i = get_local_id(0);
                const SIZE_T group = get_group_id(0);
                const SIZE_T workSize = get_local_size(0);
                const SIZE_T index = group*workSize + i;
                ${ctype} absGk;
                ${ctype} temp;
                absGk.x = G_k[index].x - Gp_k[index].x;
                absGk.y = G_k[index].y - Gp_k[index].y;
        
                tempArray[index] = absGk.x * absGk.x + absGk.y * absGk.y;
            }
    
            KERNEL void makegp_k(
                GLOBAL_MEM ${ctype} *Gp_k,
                GLOBAL_MEM double *gp_k)
            {
                const SIZE_T i = get_local_id(0);
                const SIZE_T group = get_group_id(0);
                const SIZE_T workSize = get_local_size(0);
                const SIZE_T index = group*workSize + i;
        
                gp_k[index] = Gp_k[index].x;
            }
    
            KERNEL void makeD(
                GLOBAL_MEM double *gp_k,
                GLOBAL_MEM ${ctype} *g_k,
                GLOBAL_MEM double *B,
                GLOBAL_MEM double *D)
            {
                const SIZE_T i = get_local_id(0);
                const SIZE_T group = get_group_id(0);
                const SIZE_T workSize = get_local_size(0);
                const SIZE_T index = group*workSize + i;
        
                D[index] = gp_k[index] - g_k[index].x + (B[1]/B[0]) * D[index];
            }
    
            KERNEL void makegpp_k(
                GLOBAL_MEM ${ctype} *g_k,
                GLOBAL_MEM double *D,
                GLOBAL_MEM double *gpp_k)
            {
                const SIZE_T i = get_local_id(0);
                const SIZE_T group = get_group_id(0);
                const SIZE_T workSize = get_local_size(0);
                const SIZE_T index = group*workSize + i;
        
                const double h = ${h};
                gpp_k[index] = g_k[index].x + h * D[index];
            }
    
            KERNEL void makeg_k(
                GLOBAL_MEM ${ctype} *g_k,
                GLOBAL_MEM double *gpp_k,
                GLOBAL_MEM double *support)
            {
                const SIZE_T i = get_local_id(0);
                const SIZE_T group = get_group_id(0);
                const SIZE_T workSize = get_local_size(0);
                const SIZE_T index = group*workSize + i;
        
                g_k[index].x = gpp_k[index] * support[index];
                g_k[index].y = 0;
            }

            """,render_kwds=dict(
                ctype=dtypes.ctype(dtype[1]),
                polar=functions.polar(dtype[0]),
                h=self.h))
        
        
        while k < KMAX:
            #G_k = fftpack.fftn(g_k)                           #step 1
            cfft(G_k_dev,g_k_dev)
            #Gp_k = mag * sp.exp(1j *sp.angle(G_k))            #step 2
            prg1.makeGp_k(mag_dev,G_k_dev,Gp_k_dev,local_size=WORK_SIZE,global_size=mag.size)
            #B['bk'] = X.size**-2 * (sp.absolute(G_k - Gp_k)**2).sum()   #equation 11
            prg1.makeB_kHelper(G_k_dev,Gp_k_dev,tempArray_dev,local_size=WORK_SIZE,global_size=mag.size)
            Bk = X.size**-2 * tempArray_dev.get().sum()
            B_dev = thr.to_device(sp.array([Bk1,Bk]).astype(dtype[0]))    
            #gp_k = fftpack.ifftn(Gp_k).real                   #step 3
            cfft(Gp_k_dev,Gp_k_dev,1)
            prg1.makegp_k(Gp_k_dev,gp_k_dev,local_size=WORK_SIZE,global_size=mag.size)
            #D = gp_k - g_k + (B['bk']/B['bk-1'])*D              #equation 38
            prg1.makeD(gp_k_dev,g_k_dev,B_dev,D_dev,local_size=WORK_SIZE,global_size=mag.size)
            #gpp_k = g_k + h*D                                 #equation 36
            prg1.makegpp_k(g_k_dev,D_dev,gpp_k_dev,local_size=WORK_SIZE,global_size=mag.size)
            #g_k = gpp_k * support                             #step 4 (equation 35)
            prg1.makeg_k(g_k_dev,gpp_k_dev,support_dev,local_size=WORK_SIZE,global_size=mag.size)
            Bk1 = Bk   
            if X.size**2*Bk < .0000001:                    #B_k is the error in fourier domain
                break
            k = k + 1
            if k == 10:
                g_k = g_k_dev.get().real
                ax = fig.add_subplot(223,projection='3d')
                ax.set_zlim(g_k.min(),g_k.max())
                ax.plot_surface(X,Y,g_k,rstride=1,cstride=1,cmap=cm.Accent_r,linewidth=0,antialiased=False)
                plt.title("10 iterations")
        
        g_k = g_k_dev.get().real
        #print("done")
        
        ax2 = fig.add_subplot(224, projection='3d')
        ax2.set_zlim(g_k.min(),g_k.max())
        ax2.plot_surface(X, Y, g_k,rstride=1, cstride=1, cmap=cm.Accent_r,
                               linewidth=0, antialiased=False)
        plt.title("{0} iterations".format(k))
        sfile = BytesIO()
        plt.savefig(sfile,format="svg")
        plt.close()
        return b''.join(sfile.getvalue().split(os.linesep.encode()))
        
        
c = Conjugate_Gradient(signal,n,xLower,xUpper,yLower,yUpper,h)
print(c.run())
