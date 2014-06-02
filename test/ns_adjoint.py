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
        
Ni, Nj = 100, 20
# Ni, Nj = 200, 40
theta = linspace(0, pi, Ni/2+1)
r = 15 + 5 * sin(linspace(-np.pi/2, np.pi/2, Nj+1))
r, theta = meshgrid(r, theta)
x, y = r * sin(theta), r * cos(theta)

dx = 15 * 2 * pi / Ni
y0, y1 = y[0,:], y[-1,:]
y0, x0 = meshgrid(y0, dx * arange(-Ni/4, 0))
y1, x1 = meshgrid(y1, -dx * arange(1, 1 + Ni/4))

x, y = vstack([x0, x, x1]), vstack([y0, y, y1])

geo = geo2d([x, y])

data = np.load('bend{0}x{1}-jacobian.npz'.format(Ni, Nj))
Jt = sp.csr_matrix((data['x'], data['y'], data['z']), shape=(4*Ni*Nj, 4*Ni*Nj))

xc, yc = base(geo.xyc)
g = np.exp(-(xc+10)**2/2 - (yc+15)**2/2)
gf = np.zeros([4, Ni, Nj])
gf[2] = g

#discrete adjoint
whd = splinalg.spsolve(Jt, np.ravel(gf), use_umfpack=False)
whd = whd.reshape([4,Ni,Nj])

flow = np.load('bend{0}x{1}-flow.npz'.format(Ni, Nj))
mu, al = 1, 1
w = np.array([flow['w1'], flow['w2'], flow['w3'], flow['w4']])
w_i = (w[:,1:,1:-1] + w[:,:-1,1:-1])/2
w_j = (w[:,1:-1,1:] + w[:,1:-1,:-1])/2

def grad_dual(phi, geo):
    '''
    Gradient on the nodes assuming zero boundary conditions
    '''
    dxy_i = 0.5 * (geo.dxy_i[:,1:,:] + geo.dxy_i[:,:-1,:])
    dxy_j = 0.5 * (geo.dxy_j[:,:,1:] + geo.dxy_j[:,:,:-1])
    phi_i = array([dxy_i[1] * phi, dxy_i[0] * -phi])
    phi_j = array([dxy_j[1] * -phi, dxy_j[0] * phi])

    grad_phi = zeros(geo.xy.shape)
    grad_phi[:,:-1,:-1] += phi_i + phi_j
    grad_phi[:,1:,:-1] += -phi_i + phi_j
    grad_phi[:,:-1,1:] += phi_i - phi_j
    grad_phi[:,1:,1:] += -phi_i - phi_j

    area = zeros(geo.xy.shape[1:])
    area[:-1,:-1] += geo.area
    area[1:,:-1] += geo.area
    area[:-1,1:] += geo.area
    area[1:,1:] += geo.area

    return 2 * grad_phi / area

g_w1 = grad_dual(w[0,1:-1,1:-1], geo)
g_w2 = grad_dual(w[1,1:-1,1:-1], geo)
g_w3 = grad_dual(w[2,1:-1,1:-1], geo)
g_w4 = grad_dual(w[3,1:-1,1:-1], geo)
g_w = base(array([g_w1, g_w2, g_w3, g_w4]))
g_w_j = (g_w[:,:,1:,:] + g_w[:,:,:-1,:])/2
g_w_i = (g_w[:,:,:,1:] + g_w[:,:,:,:-1])/2
g_w_i[:,[0,-1],:] = 0

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
    rhoU_n = sum(wh[1:3,1:-1,0] * geo.normal_j[:,:,0], 0)
    wh[1:3,1:-1,0] -= 2 * rhoU_n * geo.normal_j[:,:,0]

    wh[:,:,-1] = wh[:,:,-2]
    rhoU_n = sum(wh[1:3,1:-1,-1] * geo.normal_j[:,:,-1], 0)
    wh[1:3,1:-1,-1] -= 2 * rhoU_n * geo.normal_j[:,:,-1]

    return wh
    w1, w2, w3, w4 = w[0], w[1], w[2], w[3]
    w1x, w2x, w3x, w4x = g_w[0,0], g_w[1,0], g_w[2,0], g_w[3,0]
    w1y, w2y, w3y, w4y = g_w[0,1], g_w[1,1], g_w[2,1], g_w[3,1]
    one = np.ones(w[0].shape)
    zero = one*0
    g = one*1.4
    
    A = np.array([[[zero,
         -4*mu*w2x/(3*w1**2) + 2*mu*w3y/(3*w1**2) + 8*mu*(w1*w2x - w1x*w2)/(3*w1**3) - 4*mu*(w1*w3y - w1y*w3)/(3*w1**3) - w2**2/w1**2 + (g - 1)*(w2**2 + w3**2)/(2*w1**2),
         -mu*w2y/w1**2 - mu*w3x/w1**2 + 2*mu*(w1*w2y - w1y*w2)/w1**3 + 2*mu*(w1*w3x - w1x*w3)/w1**3 - w2*w3/w1**2,
         al*(-2*w2*w2x/w1**3 + 3*w1x*w2**2/w1**4) + al*(-2*w3*w3x/w1**3 + 3*w1x*w3**2/w1**4) - al*w4x/w1**2 + 2*al*(w1*w4x - w1x*w4)/w1**3 - w2*(4*mu*w2x/(3*w1**2) - 2*mu*w3y/(3*w1**2) - 8*mu*(w1*w2x - w1x*w2)/(3*w1**3) + 4*mu*(w1*w3y - w1y*w3)/(3*w1**3))/w1 - w3*(mu*w2y/w1**2 + mu*w3x/w1**2 - 2*mu*(w1*w2y - w1y*w2)/w1**3 - 2*mu*(w1*w3x - w1x*w3)/w1**3)/w1 - w2*(g*w4 - (g - 1)*(w2**2 + w3**2)/(2*w1))/w1**2 + w2*(4*mu*(w1*w2x - w1x*w2)/(3*w1**2) - 2*mu*(w1*w3y - w1y*w3)/(3*w1**2))/w1**2 + w3*(mu*(w1*w2y - w1y*w2)/w1**2 + mu*(w1*w3x - w1x*w3)/w1**2)/w1**2 + w2*(g - 1)*(w2**2 + w3**2)/(2*w1**3)],
        [one, 4*mu*w1x/(3*w1**2) - w2*(g - 1)/w1 + 2*w2/w1,
         mu*w1y/w1**2 + w3/w1,
         al*(w2x/w1**2 - 2*w1x*w2/w1**3) + 4*mu*w1x*w2/(3*w1**3) + mu*w1y*w3/w1**3 + (g*w4 - (g - 1)*(w2**2 + w3**2)/(2*w1))/w1 - (4*mu*(w1*w2x - w1x*w2)/(3*w1**2) - 2*mu*(w1*w3y - w1y*w3)/(3*w1**2))/w1 - w2**2*(g - 1)/w1**2],
        [zero, -2*mu*w1y/(3*w1**2) - w3*(g - 1)/w1, mu*w1x/w1**2 + w2/w1,
         al*(w3x/w1**2 - 2*w1x*w3/w1**3) + mu*w1x*w3/w1**3 - 2*mu*w1y*w2/(3*w1**3) - (mu*(w1*w2y - w1y*w2)/w1**2 + mu*(w1*w3x - w1x*w3)/w1**2)/w1 - w2*w3*(g - 1)/w1**2],
        [zero, g - 1, zero, al*w1x/w1**2 + g*w2/w1]],

       [[zero,
         -mu*w2y/w1**2 - mu*w3x/w1**2 + 2*mu*(w1*w2y - w1y*w2)/w1**3 + 2*mu*(w1*w3x - w1x*w3)/w1**3 - w2*w3/w1**2,
         2*mu*w2x/(3*w1**2) - 4*mu*w3y/(3*w1**2) - 4*mu*(w1*w2x - w1x*w2)/(3*w1**3) + 8*mu*(w1*w3y - w1y*w3)/(3*w1**3) - w3**2/w1**2 + (g - 1)*(w2**2 + w3**2)/(2*w1**2),
         al*(-2*w2*w2y/w1**3 + 3*w1y*w2**2/w1**4) + al*(-2*w3*w3y/w1**3 + 3*w1y*w3**2/w1**4) - al*w4y/w1**2 + 2*al*(w1*w4y - w1y*w4)/w1**3 - w2*(mu*w2y/w1**2 + mu*w3x/w1**2 - 2*mu*(w1*w2y - w1y*w2)/w1**3 - 2*mu*(w1*w3x - w1x*w3)/w1**3)/w1 - w3*(-2*mu*w2x/(3*w1**2) + 4*mu*w3y/(3*w1**2) + 4*mu*(w1*w2x - w1x*w2)/(3*w1**3) - 8*mu*(w1*w3y - w1y*w3)/(3*w1**3))/w1 + w2*(mu*(w1*w2y - w1y*w2)/w1**2 + mu*(w1*w3x - w1x*w3)/w1**2)/w1**2 - w3*(g*w4 - (g - 1)*(w2**2 + w3**2)/(2*w1))/w1**2 + w3*(-2*mu*(w1*w2x - w1x*w2)/(3*w1**2) + 4*mu*(w1*w3y - w1y*w3)/(3*w1**2))/w1**2 + w3*(g - 1)*(w2**2 + w3**2)/(2*w1**3)],
        [zero, mu*w1y/w1**2 + w3/w1, -2*mu*w1x/(3*w1**2) - w2*(g - 1)/w1,
         al*(w2y/w1**2 - 2*w1y*w2/w1**3) - 2*mu*w1x*w3/(3*w1**3) + mu*w1y*w2/w1**3 - (mu*(w1*w2y - w1y*w2)/w1**2 + mu*(w1*w3x - w1x*w3)/w1**2)/w1 - w2*w3*(g - 1)/w1**2],
        [one, mu*w1x/w1**2 + w2/w1,
         4*mu*w1y/(3*w1**2) - w3*(g - 1)/w1 + 2*w3/w1,
         al*(w3y/w1**2 - 2*w1y*w3/w1**3) + mu*w1x*w2/w1**3 + 4*mu*w1y*w3/(3*w1**3) + (g*w4 - (g - 1)*(w2**2 + w3**2)/(2*w1))/w1 - (-2*mu*(w1*w2x - w1x*w2)/(3*w1**2) + 4*mu*(w1*w3y - w1y*w3)/(3*w1**2))/w1 - w3**2*(g - 1)/w1**2],
        [zero, zero, g - 1, al*w1y/w1**2 + g*w3/w1]]])     

    return A

def visc_jacobian(w):
    w1, w2, w3, w4 = w[0], w[1], w[2], w[3]
    one = np.ones(w[0].shape)
    zero = one*0
    g = one*1.4
    D = np.array([[[[zero, -4*mu*w2/(3*w1**2), -mu*w3/w1**2,
          -al*w4/w1**2 + al*w2**2/w1**3 + al*w3**2/w1**3 - 4*mu*w2**2/(3*w1**3) - mu*w3**2/w1**3],
         [zero, 4*mu/(3*w1), zero, -al*w2/w1**2 + 4*mu*w2/(3*w1**2)],
         [zero, zero, mu/w1, -al*w3/w1**2 + mu*w3/w1**2],
         [zero, zero, zero, al/w1]],
        [[zero, -mu*w3/w1**2, 2*mu*w2/(3*w1**2), -mu*w2*w3/(3*w1**3)],
         [zero, zero, -2*mu/(3*w1), -2*mu*w3/(3*w1**2)],
         [zero, mu/w1, zero, mu*w2/w1**2],
         [zero, zero, zero, zero]]],
       [[[zero, 2*mu*w3/(3*w1**2), -mu*w2/w1**2, -mu*w2*w3/(3*w1**3)],
         [zero, zero, mu/w1, mu*w3/w1**2],
         [zero, -2*mu/(3*w1), zero, -2*mu*w2/(3*w1**2)],
         [zero, zero, zero, zero]],
        [[zero, -mu*w2/w1**2, -4*mu*w3/(3*w1**2),
          -al*w4/w1**2 + al*w2**2/w1**3 + al*w3**2/w1**3 - mu*w2**2/w1**3 - 4*mu*w3**2/(3*w1**3)],
         [zero, mu/w1, zero, -al*w2/w1**2 + mu*w2/w1**2],
         [zero, zero, 4*mu/(3*w1), -al*w3/w1**2 + 4*mu*w3/(3*w1**2)],
         [zero, zero, zero, al/w1]]]])
    return D

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

    #inviscid
    g_wh_i = (wh_ext[:,1:,1:-1] - wh_ext[:,:-1,1:-1])/2
    g_wh_j = (wh_ext[:,1:-1,1:] - wh_ext[:,1:-1,:-1])/2
    A_i = jacobian(w_i, g_w_i)
    A_j = jacobian(w_j, g_w_j)
    F_i = sum(g_wh_i * A_i[0], axis=1)
    F_j = sum(g_wh_j * A_j[0], axis=1)
    G_i = sum(g_wh_i * A_i[1], axis=1)
    G_j = sum(g_wh_j * A_j[1], axis=1)
    Fi = + F_i * geo.dxy_i[1] - G_i * geo.dxy_i[0]
    Fj = - F_j * geo.dxy_j[1] + G_j * geo.dxy_j[0]
    #viscous
    #UNDERSTAND HOWTF THE GRADIENT IS COMPUTED
    g_wh1 = grad_dual(wh_ext[0,1:-1,1:-1], geo)
    g_wh2 = grad_dual(wh_ext[1,1:-1,1:-1], geo)
    g_wh3 = grad_dual(wh_ext[2,1:-1,1:-1], geo)
    g_wh4 = grad_dual(wh_ext[3,1:-1,1:-1], geo)
    g_wh = array([g_w1, g_wh2, g_wh3, g_wh4])
    g_wh_j = (g_wh[:,:,1:,:] + g_wh[:,:,:-1,:])/2
    g_wh_i = (g_wh[:,:,:,1:] + g_wh[:,:,:,:-1])/2
    g_wh_i[:,:,[0,-1],:] = 0
    D_i = visc_jacobian(w_i)
    D_j = visc_jacobian(w_j)
    F_i = sum(g_wh_i[:,0] * D_i[0,0] + g_wh_i[:,1] * D_i[0, 1], axis=1)
    F_j = sum(g_wh_j[:,0] * D_j[0,0] + g_wh_j[:,1] * D_j[0, 1], axis=1)
    G_i = sum(g_wh_i[:,0] * D_i[1,0] + g_wh_i[:,1] * D_i[1, 1], axis=1)
    G_j = sum(g_wh_j[:,0] * D_j[1,0] + g_wh_j[:,1] * D_j[1, 1], axis=1)
    Fi_v = + F_i * geo.dxy_i[1] - G_i * geo.dxy_i[0]
    Fj_v = - F_j * geo.dxy_j[1] + G_j * geo.dxy_j[0]
    # sponge
    w1, w2, w3, w4 = w[0], w[1], w[2], w[3]
    g = 1.4
    U2 = (w2**2+w3**2)/w1**2
    p = (g-1)*(w4 - w1*U2/2)
    c = sqrt(1.4*p/w1)
    Fi_s, Fj_s = sponge_flux(adarray(c), wh_ext, geo)
    Fi_v[:5,:] += 0.5 * Fi_s[:5,:]
    Fi_v[-5:,:] += 0.5 * Fi_s[-5:,:]
    # residual 
    divF = (Fi[:,1:,:] + Fi[:,:-1,:] + Fj[:,:,1:] + Fj[:,:,:-1]) / geo.area
    divF += (Fi_v[:,1:,:] - Fi_v[:,:-1,:] + Fj_v[:,:,1:] - Fj_v[:,:,:-1]) / geo.area

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
var = 0
lim = [min(wh[var,:,:].min(), whd[var,:,:].min()), max(wh[var,:,:].max(),wh[var:,:,:].max())]

im=axes[0].contourf(xc, yc, whd[var,:,:], 100, vmin=lim[0], vmax=lim[1])
axes[0].set_title('discrete adjoint')
axes[0].axis('scaled')

im=axes[1].contourf(xc, yc, wh[var,1:-1,1:-1], 100, vmin=lim[0], vmax=lim[1])
axes[1].set_title('continuous adjoint')
axes[1].axis('scaled')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
show()

flow = np.load('bend{0}x{1}-flow-perturbed.npz'.format(Ni, Nj))
wp = np.array([flow['w1'], flow['w2'], flow['w3'], flow['w4']])
pf = np.zeros([4,Ni,Nj])
perturbation(base(geo.xyc), pf)
print 'objective delta ', sum((w[:,1:-1,1:-1]-wp[:,1:-1,1:-1])*gf)
print 'discrete adjoint prediction', sum(whd*pf)
print 'continuous adjoint prediction', sum(wh[:,1:-1,1:-1]*pf)

