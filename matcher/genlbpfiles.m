function genlbpfiles(mode)
	if strcmp(mode, 'training')
		genlbpfilesfortraining();
	elseif strcmp(mode, 'blur')
		genlbpfilesforblur();
	end
end
function genlbpfilesfortraining()
	patches = getDivide();
	trainingset = dir('~/YaleB/training/');
	for i=trainingset'
		tic;
		if i.name == '.', continue; end
		image = imread(['~/YaleB/training/', i.name]);
		h59x8000 = callbp(patches, image);
		save(['../lbpdata/mat/', i.name, '59x8000.mat'], 'h59x8000');
		%savecsv(i.name, h59x8000);
		toc;
	end
end
function genlbpfilesforblur()
	patches = getDivide();
	trainingset = dir('~/YaleB/blur/');
	for i=trainingset'
		tic;
		if i.name == '.', continue; end
		image = imread(['~/YaleB/blur/', i.name]);
		h59x8000 = callbp(patches, image);
		save(['../lbpdata/blurmat/', i.name, '59x8000.mat'], 'h59x8000');
		toc;
	end
end

function h = callbp(patches, image)
	mapping = getmapping(8, 'u2');
	h=[];
	for i = 1:size(patches)
		patch = patches(i);
		xstart = patch.xstart;
		ystart = patch.ystart;
		xend = patch.xend;
		yend = patch.yend;
		smallimage = image(xstart:xend, ystart:yend);
		h1x59 = lbp(smallimage, 1, 8, mapping, 'h');
		h = [h, h1x59'];
	end
end
