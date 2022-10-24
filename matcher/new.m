function new()
	db=database('test','hebi','','com.mysql.jdbc.Driver','jdbc:mysql://127.0.0.1:3306/test');
	for i=1:28
		ah = getahfromdb(i, db);
		bh = getbhfromdb(i, db);
		xpfinal = [];
		for j=1:836
			li = bh(:, j);%59x1
			x0 = ah'*li;
			epsilon = 0.01;
			try
				tic;
				xp = l1qc_logbarrier(x0, ah, li, epsilon, 0.01);
				toc;
				xpfinal = [xpfinal, xp];
				%disp(norm(ah*xp-li));
				disp(sprintf('%d, %d, \t%d', i, j,norm(ah*xp-li)));
			catch
				disp('===============');
				break;
			end
		end
		if(size(xpfinal, 2)==836)
			save(sprintf('../data/xps/xp%d.mat', i), 'xpfinal');
		end
	end
	close(db);
end

function test(db)
	ah = getahfromdb(2, db);
	bh = getbhfromdb(2, db);
	for j = 1:836
		li = bh(:, j);
		x0 = ah'*li;
		epsilon = 0.01;
		tic;
		xp = l1qc_logbarrier(x0, ah, li, epsilon, 10);
		toc;
		disp(norm(ah*xp-li));
		break;
	end
end
