clear
clc
close all

f = @(x) 1./(1+x.^2);

a = -5;
b = 5;
n = 20;

points = n+1;

x_vec = linspace(a,b,points);

f_vec = f(x_vec);

approx = trapz(x_vec, f_vec);

h = (b-a) / (n);

approximate = (h/2) * f_vec(1) + (h/2)*f_vec(end) + h*sum(f_vec(2:end-1));

x_vec = linspace(-5,5,21);
f_vec = f(x_vec);

value = simps(x_vec,f_vec);

% 4a)
t_vec = [2 4 6 8 10];
quadrature = zeros(1,length(t_vec));
n_vec = zeros(1,length(t_vec));
for i = 1:length(t_vec)
    funf = @(x) x.^(t_vec(i)-1) .* exp(-x);
    [quadrature(i), n_vec(i)] = quad(funf,0,10*(t_vec(i)-1));
end

gamma_truth = [1 6 120 5040 362880];

rel_error = abs(quadrature - gamma_truth) ./ gamma_truth;