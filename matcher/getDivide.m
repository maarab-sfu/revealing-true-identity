function patches = getDivide
    width = 192;
    height = 168;
    x = 6;
    y = 4;
    count = 0;
    patches=[];
    while(x<=width),
        y=4;
        while(y<=height),
            count = count+1;
            patches = [patches; getPatchesFromXY(x,y)];
            y = round(y*1.5);
            if(x*y>width*height/4),
                break;
            end
        end
        x = round(x*1.7);
    end
end
function table = getPatchesFromXY(x,y)
    table = [];
    width=192;
    height=168;
    xstart=1;
    ystart=1;
    while(xstart+x-1<=192),
        ystart=2;
        while(ystart+y-1<=168),
		tmp.xstart = xstart;
		tmp.ystart = ystart;
		tmp.xend = xstart+x-1;
		tmp.yend = ystart+y-1;
		table = [table; tmp];
            ystart = ystart+y;
        end
        xstart = xstart+x;
    end
end
