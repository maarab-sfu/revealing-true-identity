function bh = getbhfromdb(index, db, mode)
	%sql = sprintf('select * from blurlbp where class=%d', index-1);
	sql = sprintf('select * from bh4 where class=%d', index-1);
	cursor = exec(db, sql);
	cursor = fetch(cursor);
	dat = cursor.Data;
	bh = zeros(18, 836);
	for i=1:18
		for j=1:836
			bh(i,j) = dat{i, j+2};
		end
	end
	close(cursor);
end
