function [xp, up, niter] = l1qc_newton(x0, u0, A, b, epsilon, tau, newtontol, newtonmaxiter) 
	alpha = 0.01;
	beta = 0.5;  
	AtA = A'*A;
	x = x0;
	u = u0;
	r = A*x-b;
	fu1 = x - u;
	fu2 = -x - u;
	fe = 1/2*(r'*r - epsilon^2);
	f = sum(u) - (1/tau)*(sum(log(-fu1)) + sum(log(-fu2)) + log(-fe));
	niter = 0;
	done = 0;
	while (~done)
		atr = A'*r;
		ntgz = 1./fu1 - 1./fu2 + 1/fe*atr;
		ntgu = -tau - 1./fu1 - 1./fu2;
		gradf = -(1/tau)*[ntgz; ntgu];
		sig11 = 1./fu1.^2 + 1./fu2.^2;
		sig12 = -1./fu1.^2 + 1./fu2.^2;
		sigx = sig11 - sig12.^2./sig11;
		w1p = ntgz - sig12./sig11.*ntgu;
		H11p = diag(sigx) - (1/fe)*AtA + (1/fe)^2*atr*atr';
		opts.POSDEF = true; opts.SYM = true;
		[dx,hcond] = linsolve(H11p, w1p, opts);
		if (hcond < 1e-14)
			disp('Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)');
			xp = x;  up = u;
			return
		end
		Adx = A*dx;
		du = (1./sig11).*ntgu - (sig12./sig11).*dx;  
		ifu1 = find((dx-du) > 0); ifu2 = find((-dx-du) > 0);
		aqe = Adx'*Adx;   bqe = 2*r'*Adx;   cqe = r'*r - epsilon^2;
		smax = min(1,min([...
			-fu1(ifu1)./(dx(ifu1)-du(ifu1)); -fu2(ifu2)./(-dx(ifu2)-du(ifu2)); ...
			(-bqe+sqrt(bqe^2-4*aqe*cqe))/(2*aqe)
		]));
		s = (0.99)*smax;
		suffdec = 0;
		backiter = 0;
		while (~suffdec)
			xp = x + s*dx;  up = u + s*du;  rp = r + s*Adx;
			fu1p = xp - up;  fu2p = -xp - up;  fep = 1/2*(rp'*rp - epsilon^2);
			fp = sum(up) - (1/tau)*(sum(log(-fu1p)) + sum(log(-fu2p)) + log(-fep));
			flin = f + alpha*s*(gradf'*[dx; du]);
			suffdec = (fp <= flin);
			s = beta*s;
			backiter = backiter + 1;
			if (backiter > 32)
				disp('Stuck on backtracking line search, returning previous iterate.  (See Section 4 of notes for more information.)');
				xp = x;  up = u;
				return
			end
		end
		x = xp; u = up;  r = rp;
		fu1 = fu1p;  fu2 = fu2p;  fe = fep;  f = fp;
		lambda2 = -(gradf'*[dx; du]);
		stepsize = s*norm([dx; du]);
		niter = niter + 1;
		done = (lambda2/2 < newtontol) | (niter >= newtonmaxiter);
		%disp(sprintf('%d, %f', niter, lambda2/2));
	end
