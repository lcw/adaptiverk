using StaticArrays, LinearAlgebra, Plots
using Base: @kwdef

@kwdef struct RK45Options
  relativetolerance = 1e-3
end

"""   usage (time, y) = rk45(f, tspan, y0, options, varargin...)
 
 Input  - f        is the ode function
        - varargin is the parmaters that will be passed to f after t and y
        - tspan    is [t0 tfinal]
        - y0       is the initial conditions y(t0)
        - options  is the options
 Output - time     is the vector of times that y is evaluated at
        - y        is the computed solution y(j,:) at time t(j)"""
function rk45(f, tspan, y0, options, varargin...)
  T = eltype(y0)

  A = @SMatrix T[    0            0           0           0         0         0     0;
                     1//5         0           0           0         0         0     0;
                     3//40        9//40       0           0         0         0     0;
                    44//45      -56//15      32//9        0         0         0     0;
                 19372//6561 -25360//2187 64448//6561 -212//729     0         0     0;
                  9017//3168   -355//33   46732//5247   49//176 -5103//18656  0     0;
                    35//384       0         500//1113  125//192 -2187//6784  11//84 0]

  c = @SMatrix T[0 1//5 3//10 4//5 8//9 1 1]
  b = @SMatrix T[35//384 0 500//1113 125//192 -2187//6784 11//84 0;
                 5179//57600 0 7571//16695 393//640 -92097//339200 187//2100 1//40]
  e = b[1,:] - b[2,:]
  p = 5

  t0, tfinal = tspan

  alphamax = 2
  alphamin = 0.5

  tau = options.relativetolerance

  C = 0.8
  S = 0.8
  h0 = 1e-13

  ts = Vector{typeof(t0)}(undef,0)
  ys = Vector{typeof(y0)}(undef,0)
  es = Vector{eltype(y0)}(undef,0)

  ta = Vector{typeof(t0)}(undef,0)
  ha = Vector{typeof(t0)}(undef,0)

  tr = Vector{typeof(t0)}(undef,0)
  hr = Vector{typeof(t0)}(undef,0)

  append!(ts, (t0,))
  append!(ys, (copy(y0),))
  append!(es, (zero(eltype(y0)),))

  # Find the initial time step
  f0 = f(t0,y0,varargin...)
  # Save for later in the rk loop
  K1 = f0

  t1 = t0+h0
  y1 = y0+h0*f0
  f1 = f(t1,y1,varargin...)

  delta = norm(f1-f0,Inf)/norm(y1-y0,Inf)
  h = C/delta

  done = false
  step = 1
  while(!done)
    # decrease timestep if we are going to stop this time
    h, mini = findmin((tfinal-t0, h))
    done = (mini == 1) ? true : false

    y1 = y0 + h*A[2,1]*K1
    t1 = t0 + c[2]*h
    K2 = f(t1, y1)

    y1 = y0 + h*(A[3,1]*K1 + A[3,2]*K2)
    t1 = t0 + c[3]*h
    K3 = f(t1, y1)

    y1 = y0 + h*(A[4,1]*K1 + A[4,2]*K2 + A[4,3]*K3)
    t1 = t0 + c[4]*h
    K4 = f(t1, y1)

    y1 = y0 + h*(A[5,1]*K1 + A[5,2]*K2 + A[5,3]*K3 + A[5,4]*K4)
    t1 = t0 + c[5]*h
    K5 = f(t1, y1)

    y1 = y0 + h*(A[6,1]*K1 + A[6,2]*K2 + A[6,3]*K3 + A[6,4]*K4 + A[6,5]*K5)
    t1 = t0 + c[6]*h
    K6 = f(t1, y1)

    y1 = y0 + h*(A[7,1]*K1 + A[7,2]*K2 + A[7,3]*K3 + A[7,4]*K4 + A[7,5]*K5 + A[7,6]*K6)
    t1 = t0 + c[7]*h
    K7 = f(t1, y1)

    ye = h*(e[1]*K1 + e[2]*K2 + e[3]*K3 + e[4]*K4 + e[5]*K5 + e[6]*K6 + e[7]*K7)

    t1 = t0 + h

    # compute est
    est = max(norm(ye,Inf)/norm(y1,Inf),eps(t0))

    # compute alpha
    alpha = max(alphamin, min(alphamax, (tau/est)^(1/p)))

    if(alpha<1)
      # reject
      append!(tr, (t0,))
      append!(hr, (h,))

      notdone = true
    else
      # accept
      append!(ta, (t0,))
      append!(ha, (h,))

      # set local variables for the next timestep
      t0 = t1
      y0 = y1

      # set ouput variables;
      step = step+1
      append!(ts, (t0,))
      append!(ys, (copy(y0),))
      append!(es, (est,))

      # save a step
      K1 = K7
    end

    # Set the next h
    hmin = 16*eps(t0)*abs(t0)
    h = max(S*alpha*h, hmin)
  end

  (ts, ys, es, ta, ha, tr, hr)
end

function f(t, y)
  T = eltype(y)
  @SVector [one(T) + y[1]^2*y[2] - 4*y[1],
            3y[1]-y[1]^2*y[2]]
end

options = RK45Options(relativetolerance=1e-13)
y0 = @SVector [1.01, 3.0]

ts, ys, es, ta, ha, tr, hr = rk45(f, [0.0, 20.0], y0, options)

a = [y[1] for y in ys]
b = [y[2] for y in ys]
plot(ts, [a, b])

plot(ta, ha)
scatter!(tr, hr)
yaxis!("step size", :log10)
