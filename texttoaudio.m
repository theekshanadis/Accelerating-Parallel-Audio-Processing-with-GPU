clc;

%number of sample txt files
smpl_cnt=5;

count=1;
V=[];
%default sampling rate
Fs=44100;

for x=1:smpl_cnt
   
    v=['0' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'];
    file_name=['audioin' v(count) '.txt'];
    count=count+1;
    
    fprintf('Reading File: %s\n',file_name);
    
    A=textread(file_name,'%f');
    
    V=[V  A' ]; 
    
    disp(length(V))
    
end    

%write to audio file
audiowrite('newresult.wav',V,Fs);


