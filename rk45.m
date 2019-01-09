function [tout, yout] = rk45(ode, tspan, y0, options, varargin)
%  usage [time, y] = rk45(ode, tspan, y0, options, varargin)
%
%Input  - ode      is the ode function
%       - varargin is the parmaters that will be passed to ode after t and y
%       - tspan    is [t0 T]
%       - y0       is the initial conditions y(t0)
%       - options  is the options
%Output - time     is the vector of times that y is evaluated at
%       - y        is the computed solution y(j,:) at time t(j)
true = 1;
false = 0;

A = [[ 0           0          0           0        0          0     0]
     [ 1/5         0          0           0        0          0     0]
     [ 3/40        9/40       0           0        0          0     0]
     [ 44/45      -56/15      32/9        0        0          0     0]
     [ 19372/6561 -25360/2187 64448/6561 -212/729  0          0     0]
     [ 9017/3168  -355/33     46732/5247  49/176  -5103/18656 0     0]
     [ 35/384      0          500/1113    125/192 -2187/6784  11/84 0]];

c = [0 1/5 3/10 4/5 8/9 1 1]';
b4 = [5179/57600 0 7571/16695 393/640 -92097/339200 187/2100 1/40]';
b5 = [35/384 0 500/1113 125/192 -2187/6784 11/84 0]';

p = 4;

n = size(y0, 1);
t0 = tspan(1);
T = tspan(2);

alphamax = 2;
alphamin = 0.5;

tau = odeget(options, 'RelTol', 1e-3);


C = 0.8;
S = 0.8;
h0 = 1e-13;

t = t0;
y = y0;

tout(1,1) = t0;
yout(1,:) = y0';

% Find the initial time step
f0 = feval(ode,t0,y0,varargin{:});
t1 = t0+h0;
y1 = y0+h0*f0;
f1 = feval(ode,t1,y1,varargin{:});

f = zeros(n,7);
f(:,1) = f0;

delta = norm(f1-f0,2)/norm(y1-y0,2);
h = C/delta;

notdone = true;
step = 1;
while(notdone)
  % decrease timestep if we are going to stop this time
  [h, mini] = min([T-t, h]);
  notdone = mini-1;

  % compute y4 y5
  f(:,2) = feval(ode,t+h*c(2),y+h*f*A(2,:)',varargin{:});
  f(:,3) = feval(ode,t+h*c(3),y+h*f*A(3,:)',varargin{:});
  f(:,4) = feval(ode,t+h*c(4),y+h*f*A(4,:)',varargin{:});
  f(:,5) = feval(ode,t+h*c(5),y+h*f*A(5,:)',varargin{:});
  f(:,6) = feval(ode,t+h*c(6),y+h*f*A(6,:)',varargin{:});
  f(:,7) = feval(ode,t+h*c(7),y+h*f*A(7,:)',varargin{:});

  y4 = y+h*f*b4;
  y5 = y+h*f*b5;

  % compute est
  est = max(norm(y4-y5,2)/norm(y5,2),eps);

  % compute alpha
  alpha = max(alphamin, min(alphamax, (tau/est)^(1/(p+1)) ));

  if(alpha<1)
    % reject
    notdone = true;
  else
    %accept
    % set local variables for the next timestep
    t = t+h;
    y = y5;

    % set ouput variables;
    step = step+1;
    tout(step,1) = t;
    yout(step,:) = y';

    % save a step
    f(:,1) = f(:,7);
  end

  % Set the next h
  hmin = 16*eps*abs(t);
  h = max(S*alpha*h, hmin);
end

return
