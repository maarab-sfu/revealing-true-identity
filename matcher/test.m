function test()
	try
		func();
	catch MER
		disp('jglsl');
	end
end

function func()
	ME = MException('VerifyOutput:OutOfBounds', ...
             'Results are outside the allowable limits');
        throw(ME);
end
