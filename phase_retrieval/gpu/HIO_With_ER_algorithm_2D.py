import sys
signal = int(sys.argv[1])
n = int(sys.argv[2])
xLower = float(sys.argv[3])
xUpper = float(sys.argv[4])
yLower = float(sys.argv[5])
yUpper = float(sys.argv[6])
beta = float(sys.argv[7])

class HIO(object):
    def __init__(self,signal,numPoints,xLower,xUpper,yLower,yUpper,beta):
        self.signal = signal
        self.n = numPoints
        self.xLower = xLower
        self.xUpper = xUpper
        self.yLower = yLower
        self.yUpper = yUpper
        self.beta = beta
        
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
        from io import BytesIO
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib as mpl
        mpl.rcParams['xtick.labelsize'] = 6
        mpl.rcParams['ytick.labelsize'] = 6
        from matplotlib import cm
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')

        api = cluda.any_api()  
        thr = api.Thread.create()

        dtype = [sp.array([1.2]).dtype,sp.array([1j]).dtype]
        hio_iterations = 0
        er_iterations = 0
        MAX_HIO_ITERATIONS = 20  #number of HIO steps before switching to ER steps
        MAX_ER_ITERATIONS = 9    #number of ER steps before switching back to HIO
        k = 1
        KMAX = 5000  #number of total iterations

        #os.environ["PYOPENCL_COMPILER_OUTPUT"] = str(0)

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

        mag = abs(fftpack.fftn(s0))
        
        iphase = sp.random.uniform(-sp.pi,sp.pi,size=mag.size//2 - 1)
        iphase = sp.concatenate((sp.array([0]),iphase,sp.array([0]),-iphase[::-1]),axis=0).reshape(mag.shape)

        WORK_SIZE = mag.size if mag.size <= 2**8 else 2**8

        mag_dev = thr.to_device(mag.astype(dtype[0]))
        iphase_dev = thr.to_device(iphase.astype(dtype[0]))
        adft_dev = thr.empty_like(iphase.astype(dtype[1]))
        as1_dev = thr.empty_like(iphase.astype(dtype[1]))

        prg1 = thr.compile("""
        KERNEL void makeAdft(
            GLOBAL_MEM double *mag,
            GLOBAL_MEM double *iphase,
            GLOBAL_MEM ${ctype} *adft)
        {
            const SIZE_T i = get_local_id(0);
            const SIZE_T group = get_group_id(0);
            const SIZE_T workSize = get_local_size(0);
            SIZE_T index = group*workSize + i;
            adft[index] = ${polar}(mag[index],iphase[index]);
        }
        
        KERNEL void hioStep(
            GLOBAL_MEM ${ctype} *as1,
            GLOBAL_MEM ${ctype} *as2,
            GLOBAL_MEM int *support)
        {
            const SIZE_T i = get_local_id(0);
            const SIZE_T group = get_group_id(0);
            const SIZE_T workSize = get_local_size(0);
            const SIZE_T index = group*workSize + i;
            const double beta = ${beta};
            
            if(!support[index])
            {
                as2[index] = as1[index] - beta * as2[index];
            }
        }
        
         KERNEL void erStep(
            GLOBAL_MEM ${ctype} *as2,
            GLOBAL_MEM int *support)
        {
            const SIZE_T i = get_local_id(0);
            const SIZE_T group = get_group_id(0);
            const SIZE_T workSize = get_local_size(0);
            const SIZE_T index = group*workSize + i;
            
            if(!support[index])
            {
                as2[index] = 0;
            }
        }
        
        KERNEL void adftLoop(
            GLOBAL_MEM double *mag,
            GLOBAL_MEM ${ctype} *adft)
        {
            const SIZE_T i = get_local_id(0);
            const SIZE_T group = get_group_id(0);
            const SIZE_T workSize = get_local_size(0);
            const SIZE_T index = group*workSize + i;
            double arg = atan2(adft[index].y, adft[index].x); // the argument of a complex number
            adft[index] = ${polar}(mag[index],arg);           // r*e^(i*theta)
        }
        """,render_kwds=dict(
            ctype=dtypes.ctype(dtype[1]),
            polar=functions.polar(dtype[0]),
            beta=self.beta))
        prg1.makeAdft(mag_dev,iphase_dev,adft_dev,local_size=WORK_SIZE,global_size=mag.size)

        cfft = FFT(adft_dev).compile(thr)
        cfft(as1_dev,adft_dev,1) #ifft
        as2_dev = thr.to_device(as1_dev.get())


        prg1.hioStep(as1_dev,as2_dev,support_dev,local_size=WORK_SIZE,global_size=mag.size)
        as1_dev = as2_dev.copy()
        as1 = as1_dev.get().real
        ax = fig.add_subplot(222,projection='3d')
        ax.set_zlim(as1.min(),as1.max())
        ax.plot_surface(X,Y,as1,rstride=1,cstride=1,cmap=cm.Accent_r,linewidth=0,antialiased=False)
        plt.title("1 iteration")
        
        support = support_dev.get()

        while k < KMAX:
            cfft(adft_dev,as1_dev)  #fft of as1_dev gets put in adft_dev
            prg1.adftLoop(mag_dev,adft_dev,local_size=WORK_SIZE,global_size=mag.size) #mag*e^(i*argument(adft_dev))
            cfft(as2_dev,adft_dev,1)  #ifft of adft_dev gets put into as2_dev
            if hio_iterations < MAX_HIO_ITERATIONS:
                hio_iterations = hio_iterations+1
                prg1.hioStep(as1_dev,as2_dev,support_dev,local_size=WORK_SIZE,global_size=mag.size)
                as1 = as1_dev.get() #retrieve as1_dev from the gpu
                as2 = as2_dev.get()
                if (sp.absolute(as2[support==0])**2).sum() < .0000001: #stop if we have converged
                    break 
            elif er_iterations < MAX_ER_ITERATIONS:
                er_iterations = er_iterations + 1
                prg1.erStep(as2_dev,support_dev,local_size=WORK_SIZE,global_size=mag.size)
                as1 = as1_dev.get() 
                as2 = as2_dev.get()
                if (sp.absolute(as1-as2)**2).sum() < .0000001:  #if we have converged, stop
                    break
                elif er_iterations >=MAX_ER_ITERATIONS:
                    er_iterations = 0
                    hio_iterations = 0
            k = k+1
            as1_dev = as2_dev.copy()
            if k == 10:
                as1 = as1_dev.get().real
                ax = fig.add_subplot(223,projection='3d')
                ax.set_zlim(as1.min(),as1.max())
                ax.plot_surface(X,Y,as1,rstride=1,cstride=1,cmap=cm.Accent_r,linewidth=0,antialiased=False)
                plt.title("10 iterations")
        as1 = as1.real
        #print("done")

        ax2 = fig.add_subplot(224, projection='3d')
        ax2.set_zlim(as1.min(),as1.max())
        ax2.plot_surface(X, Y, as1,rstride=1, cstride=1, cmap=cm.Accent_r, linewidth=0, antialiased=False)
        plt.title("{0} iterations".format(k))

        sfile = BytesIO()
        plt.savefig(sfile,format="svg")
        plt.close()
        return b''.join(sfile.getvalue().split(os.linesep.encode()))
        
c = HIO(signal,n,xLower,xUpper,yLower,yUpper,beta)
print(c.run())