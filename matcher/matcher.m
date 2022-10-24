function [decision , file_names, score] = matcher(threshold, dirInputs, dirMapping)
    x=24;
    load(dirMapping+"\mapping.mat")
    Inputs = dir(dirInputs+"\*.png");
    N_image = length(Inputs);
    score=ones(1,N_image/2);
    dessicions=ones(N_image/2,1);
    file_names = strings(N_image/2,2);
    for i=1:2:N_image
        I=imread(dirInputs+"\"+Inputs(i).name);
        I_2=imread(dirInputs+"\"+Inputs(i+1).name);
        H1=lbp(I,2,x,mapping,'h'); 
        H11=lbp(I_2,2,x,mapping,'h'); 
        this_score=min(H1,H11);
        score((i-1)/2+1)=sum(sum(this_score));
        if(score((i-1)/2+1) > threshold)
            decision((i-1)/2+1) = 0;
        else
            decision((i-1)/2+1) = 1;
        end
        file_names((i-1)/2+1,1)=Inputs(i).name;
        file_names((i-1)/2+1,2)=Inputs(i+1).name;
    end 
end
