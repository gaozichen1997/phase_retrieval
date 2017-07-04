
import sys
signal = int(sys.argv[1])
n = int(sys.argv[2])
xLower = float(sys.argv[3])
xUpper = float(sys.argv[4])
yLower = float(sys.argv[5])
yUpper = float(sys.argv[6])

class ER(object):
    def __init__(self,signal,numPoints,xLower,xUpper,yLower,yUpper):
        self.signal = signal
        self.n = numPoints
        self.xLower = xLower
        self.xUpper = xUpper
        self.yLower = yLower
        self.yUpper = yUpper
        
    def createSignal(self,x,y):
        from scipy import sin,sqrt,pi
        if self.signal == 1:
            if x**2 + y**2 <= 1:
                return x**2 +y**2
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
        plt.style.use('ggplot')
        
        api = cluda.any_api()  
        thr = api.Thread.create()
        k = 1
        KMAX = 5000  #number of total iterations
        dtype = [sp.array([1.2]).dtype,sp.array([1j]).dtype]
        
        x = sp.linspace(self.xLower,self.xUpper,self.n,endpoint=True)
        y = sp.linspace(self.yLower,self.yUpper,self.n,endpoint=True)
        X,Y = sp.meshgrid(x,y)
        s0 = sp.array([self.createSignal(x,y) for x,y in zip(sp.ravel(X),sp.ravel(Y))]).reshape(X.shape)
        support_dev = thr.to_device(sp.array([self.createSupport(x,y) for x,y in zip(sp.ravel(X),sp.ravel(Y))]).reshape(X.shape))
        
        fig = plt.figure()
        ax = fig.add_subplot(221, projection='3d')
        ax.set_zlim(s0.min(),s0.max())
        ax.plot_surface(X, Y, s0,rstride=1, cstride=1, cmap=cm.Accent_r,
                        linewidth=0, antialiased=False)
        plt.title("Original signal")

        mag = abs(fftpack.fftn(s0))  # |F(u)|
        iphase = sp.random.uniform(-sp.pi,sp.pi,size=mag.size//2 - 1)
        iphase = sp.concatenate((sp.array([0]),iphase,sp.array([0]),-iphase[::-1]),axis=0).reshape(mag.shape)

        WORK_SIZE = mag.size if mag.size <= 2**8 else 2**8
        mag_dev = thr.to_device(mag.astype(dtype[0]))
        iphase_dev = thr.to_device(iphase.astype(dtype[0]))
        Gp_k_dev = thr.empty_like(iphase.astype(dtype[1]))
        gp_k_dev = thr.empty_like(iphase.astype(dtype[1]))
        g_k_dev = thr.empty_like(iphase.astype(dtype[0]))
        g_kTemp_dev = thr.to_device(iphase.astype(dtype[0]))
        G_k_dev = thr.empty_like(iphase.astype(dtype[1]))

        prg1 = thr.compile("""
        KERNEL void setUp(
            GLOBAL_MEM double *mag,
            GLOBAL_MEM double *iphase,
            GLOBAL_MEM ${ctype} *Gp_k)
        {
            const SIZE_T ID = get_local_id(0);
            const SIZE_T GROUP = get_group_id(0);
            const SIZE_T WORKSIZE = get_local_size(0);
            SIZE_T index = GROUP * WORKSIZE + ID;
            Gp_k[index] = ${polar}(mag[index], iphase[index]);
        }

        KERNEL void step4(
            GLOBAL_MEM ${ctype} *gp_k,
            GLOBAL_MEM double *g_k,
            GLOBAL_MEM int *support)
        {
            const SIZE_T ID = get_local_id(0);
            const SIZE_T GROUP = get_group_id(0);
            const SIZE_T WORKSIZE = get_local_size(0);
            SIZE_T index = GROUP * WORKSIZE + ID;
            
            if (support[index])
            {
                g_k[index] = gp_k[index].x;
            }
            else
            {
                g_k[index] = 0;
            }
        }
        
        KERNEL void step2(
            GLOBAL_MEM double *mag,
            GLOBAL_MEM ${ctype} *G_k,
            GLOBAL_MEM ${ctype} *Gp_k)
        {
            const SIZE_T ID = get_local_id(0);
            const SIZE_T GROUP = get_group_id(0);
            const SIZE_T WORKSIZE = get_local_size(0);
            SIZE_T index = GROUP * WORKSIZE + ID;
            
            double arg = atan2(G_k[index].y, G_k[index].x); // the arg of a complex number
            Gp_k[index] = ${polar}(mag[index], arg);
        }
        """, render_kwds = dict(
            ctype = dtypes.ctype(dtype[1]),
            polar = functions.polar(dtype[0])))

        prg1.setUp(mag_dev, iphase_dev, Gp_k_dev, local_size = WORK_SIZE, global_size = mag.size)   #step2 for first time
        cfft = FFT(Gp_k_dev).compile(thr)
        cfft(gp_k_dev, Gp_k_dev) # ifft   step 3

        prg1.step4(gp_k_dev, g_k_dev, support_dev, local_size = WORK_SIZE, global_size = mag.size)  # step 4
        g_kTemp_dev = g_k_dev.copy()

        g_k = g_k_dev.get()
        ax = fig.add_subplot(222, projection='3d')
        ax.set_zlim(g_k.min(), g_k.max())
        ax.plot_surface(X, Y, g_k, rstride=1, cstride=1, cmap=cm.Accent_r, linewidth=0, antialiased=False)
        plt.title("1 iteration")

        while k < KMAX:
            cfft(G_k_dev, g_k_dev)  # step 1
            prg1.step2(mag_dev, G_k_dev, Gp_k_dev, local_size = WORK_SIZE, global_size = mag.size) # step 2
            cfft(gp_k_dev, Gp_k_dev)  # step 3
            prg1.step4(gp_k_dev, g_kTemp_dev, support_dev, local_size = WORK_SIZE, global_size = mag.size)  # step 4

            g_k = g_k_dev.get()
            g_kTemp = g_kTemp_dev.get()
            if (sp.absolute(g_k - g_kTemp)**2).sum() < .0000001:
                break
            else:
                k = k + 1
                g_k_dev = g_kTemp_dev.copy()
                if k == 10:
                    ax = fig.add_subplot(223, projection='3d')
                    ax.set_zlim(g_kTemp.min(), g_kTemp.max())
                    ax.plot_surface(X, Y, g_kTemp, rstride=1, cstride=1, cmap=cm.Accent_r, linewidth=0, antialiased=False)
                    plt.title("10 iterations")
        g_k = g_kTemp

        ax = fig.add_subplot(224,projection='3d')
        ax.set_zlim(g_k.min(), g_k.max())
        ax.plot_surface(X, Y, g_k, rstride=1, cstride=1, cmap=cm.Accent_r, linewidth=0, antialiased=False)
        plt.title("{0} iterations".format(k))
        sfile = BytesIO()
        plt.savefig(sfile, format="svg")
        plt.close()
        return b''.join(sfile.getvalue().split(os.linesep.encode()))
c = ER(signal, n, xLower, xUpper, yLower, yUpper)
print(c.run())
        
