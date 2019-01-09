using StaticArrays, LinearAlgebra, Plots

# function f(u,t)
#   T = eltype(u)
#   @SVector [u[2], 10*(1-u[1]^2)*u[2] - u[1]]
# end

function f(u,t)
  T = eltype(u)
  @SVector [one(T) + u[1]^2*u[2] - 4*u[1],
            3u[1]-u[1]^2*u[2]]
end

function norm_sc(ys::AbstractVector{T}; sc::AbstractVector{T}) where T
  @assert length(ys) == length(sc)

  maxabs = zero(T)
  for i = 1:length(ys)
    maxabs = max(maxabs,abs(ys[i])/sc[i])
  end
  maxabs
end

function run()
  T = Float64
  A = @SMatrix T[0            0           0          0          0         0     0;
                 1//5         0           0          0          0         0     0;
                 3//40        9//40       0          0          0         0     0;
                 44//45      -56//15      32//9       0          0         0     0;
                 19372//6561 -25360//2187 64448//6561 -212//729     0         0     0;
                 9017//3168   -355//33   46732//5247   49//176 -5103//18656  0     0;
                 35//384       0         500//1113  125//192 -2187//6784  11//84 0]
  b = @SMatrix T[35//384 0 500//1113 125//192 -2187//6784 11//84 0;
                 5179//57600 0 7571//16695 393//640 -92097//339200 187//2100 1//40]
  c = @SVector T[0, 1//5, 3//10, 4//5, 8//9, 1, 1]
  e = b[1,:] - b[2,:]
  p = 5

  y0 = @SVector [T(1.01), T(3)]
  t0 = zero(T)
  e1 = e0 = zero(T)
  timeend = T(20)

  Atol = 1e-6
  Rtol = 1e-5
  sc = Atol.+abs.(y0).*Rtol

  d0 = norm_sc(y0; sc=sc)
  d1 = norm_sc(f(y0, t0); sc=sc)

  h0 = (d0 >= 1e-5 && d1 >= 1e-5) ? 0.01*(d0/d1) : 1e-6
  y1 = y0 + h0 * f(y0, t0)
  t1 = t0 + h0

  d2 = norm_sc(f(y1, t1)-f(y0, t0); sc=sc) / h0
  h1 = (max(d1, d2) <= 1e-15) ?  max(1e-6, h0 * 1e-3) : (0.01 / max(d1, d2))^1/(p+1)

  @show dt = min(100*h0, h1)

  nsteps = ceil(Int64, timeend / dt)
  dt = timeend / nsteps

  ts = Vector{typeof(t0)}(undef,nsteps+1)
  ys = Vector{typeof(y0)}(undef,nsteps+1)
  es = Vector{typeof(e0)}(undef,nsteps+1)

  ts[1] = t0
  ys[1] = copy(y0)
  es[1] = e0

  for n = 1:nsteps

    acceptstep = false

    fac = T(0.9)
    facmin = T(0.1)
    facmax = T(5)

    while !acceptstep
      y1 = y0
      t1 = t0
      K1 = f(y1, t1)

      y1 = y0 + dt*A[2,1]*K1
      t1 = t0 + c[2]*dt
      K2 = f(y1, t1)

      y1 = y0 + dt*(A[3,1]*K1 + A[3,2]*K2)
      t1 = t0 + c[3]*dt
      K3 = f(y1, t1)

      y1 = y0 + dt*(A[4,1]*K1 + A[4,2]*K2 + A[4,3]*K3)
      t1 = t0 + c[4]*dt
      K4 = f(y1, t1)

      y1 = y0 + dt*(A[5,1]*K1 + A[5,2]*K2 + A[5,3]*K3 + A[5,4]*K4)
      t1 = t0 + c[5]*dt
      K5 = f(y1, t1)
      
      y1 = y0 + dt*(A[6,1]*K1 + A[6,2]*K2 + A[6,3]*K3 + A[6,4]*K4 + A[6,5]*K5)
      t1 = t0 + c[6]*dt
      K6 = f(y1, t1)

      y1 = y0 + dt*(A[7,1]*K1 + A[7,2]*K2 + A[7,3]*K3 + A[7,4]*K4 + A[7,5]*K5 + A[7,6]*K6)
      t1 = t0 + c[7]*dt
      K7 = f(y1, t1)
      
      ye = dt*(e[1]*K1 + e[2]*K2 + e[3]*K3 + e[4]*K4 + e[5]*K5 + e[6]*K6 + e[7]*K7)

      t1 = t0 + dt
      # y1 is computed in the last stage

      k1 = one(T)
      sc = Atol.+abs.(y1).*Rtol
      e1 = norm_sc(ye, sc=sc)

      beta = e1^(-k1/p)
      dt *= min(facmax, max(facmin, fac*beta))

      if e1 <= 1
        acceptstep = true
      else
        facmax = fac = T(0.9)
      end
    end

    ys[n+1] = y1
    ts[n+1] = t1
    es[n+1] = e1

    t0 = t1
    y0 = y1
    e0 = e1
  end

  (ys, ts, es)
end

(ys, ts, es) = run()


a = [y[1] for y in ys]
b = [y[2] for y in ys]

plot(ts, [a, b])

plot(ts[2:end], es[2:end])
yaxis!("error", :log10)
