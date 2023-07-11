
function f_writeDataset_hdf5(fileName,DataIn,DataGt)
    
%     DataIn  = uint16(rand(128,128,32,10));
%     DataGt  = uint16(rand(128,128,1,10));    
    
    h5create(fileName,'/input',size(DataIn),'Datatype','single')
    h5write(fileName,'/input',DataIn)
    
    h5create(fileName,'/gt',size(DataGt),'Datatype','single')
    h5write(fileName,'/gt',DataGt)
    
    h5disp(fileName)
    
%     dataGt_rd = h5read('mydata.h5','/gt');
%     dataGt_rd = h5read('mydata.h5','/input');
    
end
