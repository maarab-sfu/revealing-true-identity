function xp = l1qc_logbarrier(x0, A, b, epsilon, lbtol)  
	mu=10;
	newtontol = lbtol;
	newtonmaxiter = 50;
	N = length(x0);	
	if (norm(A*x0-b) > epsilon)
		opts.POSDEF = true; opts.SYM = true;
		[w, hcond] = linsolve(A*A', b, opts);
		if (hcond < 1e-14)
			disp('A*At is ill-conditioned: cannot find starting point');
			xp = x0;
			return;
		end
		x0 = A'*w;
	end  
	x = x0;
	u = (0.95)*abs(x0) + (0.10)*max(abs(x0)); %what's this
	tau = max((2*N+1)/sum(abs(x0)), 1);
	lbiter = ceil((log(2*N+1)-log(lbtol)-log(tau))/log(mu));
	totaliter = 0;
	for ii = 1:lbiter
		%disp('=========================');
		[xp, up, ntiter] = l1qc_newton(x, u, A, b, epsilon, tau, newtontol, newtonmaxiter);
		totaliter = totaliter + ntiter;
		x = xp;
		u = up;
		tau = mu*tau;
	end
