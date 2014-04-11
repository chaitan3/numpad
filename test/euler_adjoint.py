import time
import sys
from pylab import *
sys.path.append('../..')
from numpad import *
from perturb import perturbation

class geo2d:
    def __init__(self, xy):
        xy = array(xy)
        self.xy = xy
        self.xyc = (xy[:,1:,1:]  + xy[:,:-1,1:] + \
                    xy[:,1:,:-1] + xy[:,:-1,:-1]) / 4

        self.dxy_i = xy[:,:,1:] - xy[:,:,:-1]
        self.dxy_j = xy[:,1:,:] - xy[:,:-1,:]

        self.L_j = sqrt(self.dxy_j[0]**2 + self.dxy_j[1]**2)
        self.L_i = sqrt(self.dxy_i[0]**2 + self.dxy_i[1]**2)
        self.normal_j = array([self.dxy_j[1] / self.L_j,
                              -self.dxy_j[0] / self.L_j])
        self.normal_i = array([self.dxy_i[1] / self.L_i,
                              -self.dxy_i[0] / self.L_i])

        self.area = self.tri_area(self.dxy_i[:,:-1,:], self.dxy_j[:,:,1:]) \
                  + self.tri_area(self.dxy_i[:,1:,:], self.dxy_j[:,:,:-1]) \

    def tri_area(self, xy0, xy1):
        return 0.5 * (xy0[1] * xy1[0] - xy0[0] * xy1[1])
        
obj = int(sys.argv[1])
if (sys.argv[2] == "f"):
    Ni, Nj = 100, 40
elif (sys.argv[2] == "ff"):
    Ni, Nj = 200, 80
else:
    Ni, Nj = 50, 20
x = np.linspace(-20,20,Ni+1)
y = np.linspace(-5, 5, Nj+1)
a = np.ones(Ni+1)
a[np.abs(x) < 10] = 1 - (1 + np.cos(x[np.abs(x) < 10] / 10 * np.pi)) * 0.1

y, x = np.meshgrid(y, x)
y *= a[:,np.newaxis]
geo = geo2d([x, y])

data = np.load('nozzle{0}x{1}-jacobian.npz'.format(Ni, Nj))
Jt = sp.csr_matrix((data['x'], data['y'], data['z']), shape=(4*Ni*Nj, 4*Ni*Nj))

xc, yc = base(geo.xyc)
g = np.exp(-(xc-5)**2/2 - (yc-0)**2/2)
gf = np.zeros([4, Ni, Nj])
gf[obj] = g

#discrete adjoint
whd = splinalg.spsolve(Jt, np.ravel(gf), use_umfpack=False)
whd = whd.reshape([4,Ni,Nj])

flow = np.load('nozzle{0}x{1}-flow.npz'.format(Ni, Nj))
w = np.array([flow['w1'], flow['w2'], flow['w3'], flow['w4']])
w_i = (w[:,1:,1:-1] + w[:,:-1,1:-1])/2
w_j = (w[:,1:-1,1:] + w[:,1:-1,:-1])/2

def extend(wh_interior, geo):
    '''
    Extend the conservative variables into ghost cells using boundary condition
    '''
    wh = zeros([4, Ni+2, Nj+2])
    wh[:,1:-1,1:-1] = wh_interior.reshape([4, Ni, Nj])
    g = 1.4

    # inlet
    w_in = w[:,1,1:-1]
    w1, w2, w3, w4 = w_in[0], w_in[1], w_in[2], w_in[3]
    wh[0:3,0,1:-1] = wh[0:3,1,1:-1]
    coeff = np.array([ -w1 + w2*sqrt(w2**2/w1**2)/(g*(g - 1)*(w4 - w2**2/(2*w1))/w1 + w2**2*(g - 1)/(2*w1**2)),
       w1*(w2*(g - 1)/w1 - 2*w2/w1) - w2*(g - 1) - sqrt(w2**2/w1**2)*(g - 1)*(w4 - w2**2/(2*w1))*(w1*(w2*(w2*(g - 1)/w1 - 2*w2/w1)/w1 - w2**2*(g - 1)/w1**2 + w2**2/w1**2)/((g - 1)*(w4 - w2**2/(2*w1))) - 1)/(g*(g - 1)*(w4 - w2**2/(2*w1))/w1 + w2**2*(g - 1)/(2*w1**2)),
       -g*w2**2/w1 + w1*(-(g*w4 - w2**2*(g - 1)/(2*w1))/w1 + w2**2*(g - 1)/w1**2) - sqrt(w2**2/w1**2)*(g - 1)*(w4 - w2**2/(2*w1))*(-g*w2/(w1*(g - 1)) + w1*(-g*w2**3/(2*w1**3) + w2*(-(g*w4 - w2**2*(g - 1)/(2*w1))/w1 + w2**2*(g - 1)/w1**2)/w1 + w2*(g*w4 - w2**2*(g - 1)/(2*w1))/w1**2 - w2**3*(g - 1)/(2*w1**3))/((g - 1)*(w4 - w2**2/(2*w1))))/(g*(g - 1)*(w4 - w2**2/(2*w1))/w1 + w2**2*(g - 1)/(2*w1**2))])


    wh[3,0,1:-1] = -(wh[0,0,1:-1]*coeff[0] + wh[1,0,1:-1]*coeff[1])/coeff[2]

    # outlet
    w_in = w[:,-2,1:-1]
    w1, w2, w3, w4 = w_in[0], w_in[1], w_in[2], w_in[3]
    wh[3,-1,1:-1] = wh[3,-2,1:-1]
    n1 = base(geo.normal_i[0,-1,:])
    n2 = base(geo.normal_i[1,-1,:])
    w23n = w2*n1 + w3*n2
    U2 = (w2**2 + w3**2)/w1**2
    coeff = np.array([g*w4/w1 + U2*(1-g)/2, -(g*(w4/w1 - U2/2)*n1*w1/w23n + w2/w1), -(g*(w4/w1 - U2/2)*n2*w1/w23n + w3/w1)])
    wh[0,-1,1:-1] = wh[3,-1,1:-1]*coeff[0]
    wh[1,-1,1:-1] = wh[3,-1,1:-1]*coeff[1]
    wh[2,-1,1:-1] = wh[3,-1,1:-1]*coeff[2]

    # walls
    wh[:,:,0] = wh[:,:,1]
    nwall = geo.normal_j[:,:,0]
    nwall = hstack([nwall[:,:1], nwall, nwall[:,-1:]])
    rhoU_n = sum(wh[1:3,:,0] * nwall, 0)
    wh[1:3,:,0] -= 2 * rhoU_n * nwall

    wh[:,:,-1] = wh[:,:,-2]
    nwall = geo.normal_j[:,:,-1]
    nwall = hstack([nwall[:,:1], nwall, nwall[:,-1:]])
    rhoU_n = sum(wh[1:3,:,-1] * nwall, 0)
    wh[1:3,:,-1] -= 2 * rhoU_n * nwall

    return wh

def jacobian(w):
    w1, w2, w3, w4 = w[0], w[1], w[2], w[3]
    one = np.ones(w1.shape)
    zero = np.zeros(w1.shape)
    g = one*1.4
    A1 = np.array([[zero, -w2**2/w1**2 + (g - 1)*(w2**2 + w3**2)/(2*w1**2),
        -w2*w3/w1**2,
        -w2*(g*w4 - (g - 1)*(w2**2 + w3**2)/(2*w1))/w1**2 + w2*(g - 1)*(w2**2 + w3**2)/(2*w1**3)],
       [one, -w2*(g - 1)/w1 + 2*w2/w1, w3/w1,
        (g*w4 - (g - 1)*(w2**2 + w3**2)/(2*w1))/w1 - w2**2*(g - 1)/w1**2],
       [zero, -w3*(g - 1)/w1, w2/w1, -w2*w3*(g - 1)/w1**2],
       [zero, g - 1, zero, g*w2/w1]])
    A2 = np.array([[zero, -w2*w3/w1**2,
        -w3**2/w1**2 + (g - 1)*(w2**2 + w3**2)/(2*w1**2),
        -w3*(g*w4 - (g - 1)*(w2**2 + w3**2)/(2*w1))/w1**2 + w3*(g - 1)*(w2**2 + w3**2)/(2*w1**3)],
       [zero, w3/w1, -w2*(g - 1)/w1, -w2*w3*(g - 1)/w1**2],
       [one, w2/w1, -w3*(g - 1)/w1 + 2*w3/w1,
        (g*w4 - (g - 1)*(w2**2 + w3**2)/(2*w1))/w1 - w3**2*(g - 1)/w1**2],
       [zero, zero, g - 1, g*w3/w1]])
    return A1, A2

def sponge_flux(c_ext, w_ext, geo):
    ci = 0.5 * (c_ext[1:,1:-1] + c_ext[:-1,1:-1])
    cj = 0.5 * (c_ext[1:-1,1:] + c_ext[1:-1,:-1])

    a = geo.area
    ai = vstack([a[:1,:], (a[1:,:] + a[:-1,:]) / 2, a[-1:,:]])
    aj = hstack([a[:,:1], (a[:,1:] + a[:,:-1]) / 2, a[:,-1:]])

    Fi = 0.5 * ci * ai * (w_ext[:,1:,1:-1] - w_ext[:,:-1,1:-1])
    Fj = 0.5 * cj * aj * (w_ext[:,1:-1,1:] - w_ext[:,1:-1,:-1])
    return Fi, Fj

def adjoint_eqns(wh, wh0, geo, dt):
    wh_ext = extend(wh, geo)
    wh_i = (wh_ext[:,1:,1:-1] + wh_ext[:,:-1,1:-1])/2
    g_wh_i = (wh_ext[:,1:,1:-1] - wh_ext[:,:-1,1:-1])/2
    wh_j = (wh_ext[:,1:-1,1:] + wh_ext[:,1:-1,:-1])/2
    g_wh_j = (wh_ext[:,1:-1,1:] - wh_ext[:,1:-1,:-1])/2
    F_i, G_i = jacobian(w_i)
    F_j, G_j = jacobian(w_j)
    F_i = sum(g_wh_i * F_i, axis=1)
    F_j = sum(g_wh_j * F_j, axis=1)
    G_i = sum(g_wh_i * G_i, axis=1)
    G_j = sum(g_wh_j * G_j, axis=1)
    Fi = + F_i * geo.dxy_i[1] - G_i * geo.dxy_i[0]
    Fj = - F_j * geo.dxy_j[1] + G_j * geo.dxy_j[0]
    # sponge
    w1, w2, w3, w4 = w[0], w[1], w[2], w[3]
    g = 1.4
    U2 = (w2**2+w3**2)/w1**2
    p = (g-1)*(w4 - w1*U2/2)
    c = sqrt(1.4*p/w1)
    Fi_s, Fj_s = sponge_flux(adarray(c), wh_ext, geo)
    #Fi[:5,:] += 0.5 * Fi_s[:5,:]
    #Fi[-5:,:] += 0.5 * Fi_s[-5:,:]
    # residual 
    divF = (Fi[:,1:,:] + Fi[:,:-1,:] + Fj[:,:,1:] + Fj[:,:,:-1]) / geo.area
    divF += 0.5*(Fi_s[:,1:,:] - Fi_s[:,:-1,:]) / geo.area

    return (wh - wh0)/dt - ravel(divF + gf) 

t, dt = 0, inf
wh = zeros([4, Ni, Nj])
wh0 = ravel(wh)

for i in range(1):
    print('i = ', i, 't = ', t)
    
    wh = solve(adjoint_eqns, wh0, args=(wh0, geo, dt), rel_tol=1E-8, abs_tol=1E-6)
    if wh._n_Newton == 1:
        break
    elif wh._n_Newton < 5:
        wh0 = wh
        dt *= 2
    elif wh._n_Newton < 10:
        wh0 = wh
    else:
        dt *= 0.5
        continue
    t += dt
    wh0.obliviate()
   
    print(adarray_count(), adstate_count())
print('Final, t = inf')
#dt = np.inf
wh = solve(adjoint_eqns, wh0, args=(wh0, geo, dt), rel_tol=1E-9, abs_tol=1E-7)
wh = base(extend(wh, geo))

fig,axes = subplots(nrows=2, ncols=1)
lim = [min(wh[0,:,:].min(), whd[0,:,:].min()), max(wh[0,:,:].max(),wh[0:,:,:].max())]

im=axes[0].contourf(xc, yc, whd[0,:,:], 100, vmin=lim[0], vmax=lim[1])
axes[0].set_title('discrete adjoint')
axes[0].axis('scaled')

im=axes[1].contourf(xc, yc, wh[0,1:-1,1:-1], 100, vmin=lim[0], vmax=lim[1])
axes[1].set_title('continuous adjoint')
axes[1].axis('scaled')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
#show()

print obj, Ni, Nj, np.linalg.norm(wh[0,1:-1,1:-1]-whd[0], 'fro')

flow = np.load('nozzle{0}x{1}-flow-perturbed.npz'.format(Ni, Nj))
wp = np.array([flow['w1'], flow['w2'], flow['w3'], flow['w4']])
pf = np.zeros([4,Ni,Nj])
perturbation(base(geo.xyc), pf)
print 'objective delta ', sum((w[:,1:-1,1:-1]-wp[:,1:-1,1:-1])*gf)
print 'discrete adjoint prediction', sum(whd*pf)
print 'continuous adjoint prediction', sum(wh[:,1:-1,1:-1]*pf)
