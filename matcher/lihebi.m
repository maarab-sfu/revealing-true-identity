function lihebi()
	db=database('test','hebi','','com.mysql.jdbc.Driver','jdbc:mysql://127.0.0.1:3306/test');
	flog = fopen('log.txt', 'w');
	for i=1:8576
		if i<1541, continue; end
		ah = getahfromdb(i, db);
		bh = getbhfromdb(i, db);
		xpfinal = [];
		for j=1:836
			disp(sprintf('%d, %d', i, j));
			fprintf(flog, '%d, %d\n', i, j);
			li = bh(:, j);%59x1
			x0 = ah'*li;
			epsilon = 0.02;
			try
			xp = l1qc_logbarrier(x0, ah, [], li, epsilon, 1e-3);
			xpfinal = [xpfinal, xp];
			catch
				disp('===============');
				fprintf(flog, '=================');
				break;
			end
		end
		disp(size(xpfinal));
		save(sprintf('../data/xps/xp%d.mat', i), 'xpfinal');
	end
	close(db);
end
