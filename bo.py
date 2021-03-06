import numpy as np
np.seterr(divide='ignore', invalid='ignore')## ignore division by 0 and nan
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from bayes_opt import BayesianOptimization


def make_curve_line():
    N = 100001
    list_x = np.linspace(0.0,np.pi,N)
    list_y = np.linspace(0.0,-2.0,N)
    return list_x, list_y

def make_curve_cycloid():
    N = 100001
    list_t = np.linspace(0.0,np.pi,N)
    list_x = list_t - np.sin(list_t)
    list_y = - 1.0 + np.cos(list_t)
    return list_x, list_y

def make_curve_points(coeff):
    len_c = len(coeff)
    list_x = np.linspace(0.0,np.pi,len_c+2)
    list_y = [0.0, -2.0]
    list_y[1:1] = coeff
    list_y = np.array(list_y)
    return list_x, list_y

def make_curve_spline(coeff):
    len_c = len(coeff)
    list_x = np.linspace(0.0,np.pi,len_c+2)
    list_y = [0.0, -2.0]
    list_y[1:1] = coeff
    list_y = np.array(list_y)
    func_spline = interp1d(list_x,list_y,kind="cubic")
    N = 100001
    list_x2 = np.linspace(0.0,np.pi,N)
    list_y2 = func_spline(list_x2)
    return list_x2, list_y2

def calc_time(list_x,list_y,g):
    time = 0.0
    len_x = len(list_x)
    list_dx = np.array([list_x[i+1]-list_x[i] for i in range(len_x-1)])
    list_dy = np.array([list_y[i+1]-list_y[i] for i in range(len_x-1)])
#    list_time = np.array([np.sqrt((1.0+(list_dy[i]/list_dx[i])**2)/(-list_y[i]))*list_dx[i] for i in range(1,len_x-1)])
#    list_time = np.array([np.sqrt((1.0+(list_dy[i]/list_dx[i])**2)/(np.abs(list_y[i])))*list_dx[i] for i in range(1,len_x-1)])
    list_time = np.array([np.sqrt(((list_dx[i])**2+(list_dy[i])**2)/(0.5*np.abs(list_y[i]+list_y[i+1]))) for i in range(0,len_x-1)])
    time = np.sum(list_time)/np.sqrt(2.0*g)
    return time

#def black_box_function(c0,c1,g):
#    list_x, list_y = make_curve_spline([c0,c1])
#    time = calc_time(list_x,list_y,g)
#    return -time

def black_box_function(c0,c1,c2,g):
    list_x, list_y = make_curve_spline([c0,c1,c2])
    time = calc_time(list_x,list_y,g)
    return -time


def main():
    g = 1.0

    list_x, list_y = make_curve_line()
    time = calc_time(list_x,list_y,g)
    print("time(line)",time)
    fig = plt.figure()
    plt.plot(list_x,list_y)
    fig.savefig("fig_curve_line.png")

    list_x, list_y = make_curve_cycloid()
    time = calc_time(list_x,list_y,g)
    print("time(cycloid)",time)
    fig = plt.figure()
    plt.plot(list_x,list_y)
    fig.savefig("fig_curve_cycloid.png")

#    coeff = [-1.25, -1.75]
#
#    list_x, list_y = make_curve_points(coeff)
#    fig = plt.figure()
#    plt.plot(list_x,list_y)
#    fig.savefig("fig_curve_points.png")
#
#    list_x, list_y = make_curve_spline(coeff)
#    time = calc_time(list_x,list_y,g)
#    print("time(spline)",time)
#    fig = plt.figure()
#    plt.plot(list_x,list_y)
#    fig.savefig("fig_curve_spline.png")

    cmin = -2.0
    cmax = -1e-6
#    pbounds = {'c0':(cmin,cmax),'c1':(cmin,cmax),'g':(g,g)}
    pbounds = {'c0':(cmin,cmax),'c1':(cmin,cmax),'c2':(cmin,cmax),'g':(g,g)}
    bo = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=987234,
    )
#    bo.maximize(n_iter=10)
#    bo.maximize(n_iter=20,acq="ucb",kappa=10)
    bo.maximize(n_iter=20,acq="ei",xi=1e-1)
#    bo.maximize(n_iter=20,acq="poi",xi=1e-1)
    print(bo.max)

#    c0s = [p['params']['c0'] for p in bo.res]
#    c1s = [p['params']['c1'] for p in bo.res]
#    print(c0s)
#    print(c1s)

    c0max = bo.max['params']['c0']
    c1max = bo.max['params']['c1']
    c2max = bo.max['params']['c2']
    print(c0max)
    print(c1max)
    print(c2max)

#    coeff = [c0max,c1max]
    coeff = [c0max,c1max,c2max]
    list_x, list_y = make_curve_spline(coeff)
    time = calc_time(list_x,list_y,g)
    print("time(opt_spline)",time)
    fig = plt.figure()
    plt.plot(list_x,list_y)
    fig.savefig("fig_curve_opt_spline.png")

if __name__ == "__main__":   
    main()
