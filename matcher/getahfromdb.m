function ah = getahfromdb(index, db) %1---8576
	%sql = sprintf('select * from matlab where class=%d', index-1);
	sql = sprintf('select * from ah4 where class=%d', index-1);
	cursor = exec(db, sql);
	cursor = fetch(cursor);
	dat = cursor.Data;
	ah = zeros(18,760);
	for i=1:18
		for j=1:760
			ah(i,j) = dat{i, j+2};
		end
	end
	close(cursor);
end
