
function emhist = f_genEmhist(max_input_photons,N_reps,pram)

  in_photons  = [1:max_input_photons]';

  emhist_100  = (repmat(in_photons,[1 100]));
  for i=1:N_reps/100
    emhist{i} = emhist_100;
  end
   


  tic
 
  fprintf('%0.4d/%0.4d',0,pram.cam_N_gainStages)
  parfor tt=1:length(emhist)
    for ii=1:pram.cam_N_gainStages
      fprintf('\b\b\b\b\b\b\b\b\b%0.4d/%0.4d',ii,pram.cam_N_gainStages)     
      emhist{tt}  = emhist{tt} + binornd(emhist{tt},pram.cam_Brnuli_alpha);

    end
  end
  fprintf('\n')
  toc

  emhist = [emhist{:}];
  pram_usedInEmhist = pram;
  
  mkdir('./_emhist')
  save(['./_emhist/emhist_' pram.dataset '_' date '.mat'],'emhist','pram_usedInEmhist')
end


