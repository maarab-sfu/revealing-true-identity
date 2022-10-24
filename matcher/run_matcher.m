% dirInputs is directory to images folder.
% dirMapping is the address of mapping file which is available in
% the this directory.
%threshold is the value for similarity score to decide between attack or bonafide 
dirInputs = ".\images\";
dirMapping = '.\';
threshold = 2.25e6;


%OUTPUTS:
%decisions is a vector containing the assigned class to each case based on threshold
%file_names contains the input images name as "n-after.png"
%scores the similarity score produced for each pair of image("n-after.png" , "n-before.png")
%Attention: These three outputs are aligned to each other. It means that
%the first score and decision in scores and decisions vectors are related to the
%first image name in the first index of file_names vector. same for other indexes. 
[decisions , file_names, scores] = matcher( threshold, dirInputs, dirMapping)

for i=1:1:length(scores)
    disp("-------------------------------------------")
    disp("for the files of:")
    disp(file_names(i,1))
    disp(file_names(i,2))
    disp("The similarity score is:")
    disp(scores(i))
    disp("The Decision based on the threshold is:")
    if(decisions(i) ==1)
        disp("Attack")
    else
        disp("Bonafide")
    end
    disp("-------------------------------------------")
end
disp("Total number of Attacks:")
disp(sum(decisions))
disp("Total number of Bonafides:")
disp(length(decisions)-sum(decisions))

figure
x=1:(length(scores));
plot(scores,'-o');
% ax = gca;
% ax.XTick = 1:size(file_names,1);
% ax.XTickLabel = file_names(:,1);
ylabel('Similarity Scores');
legend({'Inputs'},'Location','southwest')