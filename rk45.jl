using StaticArrays, LinearAlgebra, Plots
using Base: @kwdef

@kwdef struct RK45Options
  relativetolerance = 1//10^3
  absolutetolerance = 1//10^3
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

  rtol = options.relativetolerance
  atol = options.absolutetolerance

  C = 1//10
  h0 = 1//10^13

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

  estinit=1//10^4
  est_1 = est0 = one(T)
  failfactor = 2

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
    # est = max(norm(ye,Inf)/norm(y1,Inf),eps(eltype(y1)))/tau
    est = max(norm(ye./(atol+rtol*abs.(y1)),Inf),eps(eltype(y1)))

    # PI controller from Tim and Julia
    # beta2 = 4//100
    # beta1 = typeof(beta2)(1//p) - 3beta2/4
    # q11 = est^beta1
    # q = q11/(est0^beta2)
    # qmax = 10
    # qmin = 1//5
    # gamma = 9//10
    # q = max(inv(qmax),min(inv(qmin),q/gamma))

    if(est>1)
      # reject
      append!(tr, (t0,))
      append!(hr, (h,))

      # Part of the PI controller from Julia
      # h = h/min(inv(qmin),q11/gamma)

      h = h/failfactor

      notdone = true
    else
      # accept
      append!(ta, (t0,))
      append!(ha, (h,))

      # set local variables for the next timestep
      t0 = t1
      y0 = y1

      # PID controller from eq (30) using the exponents in the text after
      # eq (31) in the paper:
      #
      #     Additive Runge--Kutta schemes for convection--diffusion--reaction
      #     equations
      #
      #     Christopher A. Kennedy and Mark H. Carpenter
      #     Applied Numerical Mathematics
      #     Volume 44, Issues 1--2, January 2003, Pages 139--181
      #
      #     https://doi.org/10.1016/S0168-9274(02)00138-1
      h = (9//10) * h * est^(-0.49/(p-1)) * est0^(0.34/(p-1)) * est_1^(-0.10/(p-1))

      # PI controller from Julia
      # h = h/q

      est_1 = est0
      # est0 = max(est,estinit)
      est0 = est


      # set ouput variables;
      step = step+1
      append!(ts, (t0,))
      append!(ys, (copy(y0),))
      append!(es, (est,))

      # save a step
      K1 = K7
    end

    # Set the next h
    hmin = 16*eps(t0)
    h = max(h, hmin)
  end

  (ts, ys, es, ta, ha, tr, hr)
end

function f(t, y)
  T = eltype(y)
  @SVector [one(T) + y[1]^2*y[2] - 4*y[1],
            3y[1]-y[1]^2*y[2]]
end

options = RK45Options(relativetolerance=1e-8,
                      absolutetolerance=1e-8)
y0 = @SVector [1.01, 3.0]
tspan = (0.0, 20.0)
ts, ys, es, ta, ha, tr, hr = rk45(f, tspan, y0, options)

# a = [y[1] for y in ys]
# b = [y[2] for y in ys]
# plot(ts, [a, b])
# 
# plot(ta, ha)
# scatter!(tr, hr)
# yaxis!("step size", :log10)

using DifferentialEquations, Plots
function g(du,u,p,t)
  du[1] = one(u[1]) + u[1]^2*u[2] - 4*u[1]
  du[2] = 3u[1]-u[1]^2*u[2]
end
u0 = [1.01, 3.0]
tspan = (0.0, 20.0)
prob = ODEProblem(g, u0, tspan)
solgood = solve(prob,DP5(),reltol=1e-15,abstol=1e-15)

sol = solve(prob,DP5(),reltol=options.relativetolerance/3,
                       abstol=options.absolutetolerance/3)
 
plot(ta, ha)
plot!(sol.t[1:end-1], diff(sol.t))
scatter!(tr, hr)
yaxis!("step size", :log10)

errory = maximum(abs.(ys[end] - solgood.u[end]) ./ abs.(solgood.u[end]))
erroru = maximum(abs.(sol.u[end] - solgood.u[end]) ./ abs.(solgood.u[end]))

@show (errory, erroru)
@show length(ys)
@show length(sol.u)


# using DifferentialEquations, Plots
# function h(du,u,p,t)
#   du[1] = u[2]
#   du[2] = 10(1-u[1]^2)*u[2] - u[1]
# end
# u0 = [2.0, 0.0]
# tspan = (0.0, 20.0)
# p = ODEProblem(h, u0, tspan)
# s = solve(p,DP5(),reltol=1e-14,abstol=1e-14)
# plot(s)
