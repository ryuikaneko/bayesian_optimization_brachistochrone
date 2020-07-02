# brachistochrone curve by Bayesian optimization

- Use https://github.com/fmfn/BayesianOptimization
- Minimize the time for given curves that are obtained by spline interpolation of given points (x,y)=(0,0), (pi/N,c1), (2pi/N,c2), (3pi/N,c3), ..., ((N-1)pi/N,c{N-1}), (pi,-2)
- Parameters c1,c2,c3,...,c{N-1} are estimated by Bayesian optimization
- Optimal curve is a cycloid: x(t)=t-sin(t), y(t)=-1+cos(t), time=t(=pi at (x,y)=(pi,-2))
