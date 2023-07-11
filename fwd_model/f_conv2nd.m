% 20210128 by Dushan N. Wadduwage
% This will perform 2d convolution comparable to Matlab's conv2d, but for a
% batch of data in the form of an N-d array. The convolution will be doen
% on the first two dimentions.
%
% inputs: A : 3d array
%         B : 2d array
% output  C : 3d array


function C = f_conv2nd(A,B,shape)

  fft_Ny      = size(A,1)+size(B,1)-1;
  fft_Nx      = size(A,2)+size(B,2)-1;
  
  for i=1:size(B,5)
    for j=1:size(B,4)
      C(:,:,:,j,i) = ifft2(fft2(A,fft_Ny,fft_Nx) .* fft2(B(:,:,:,j,i),fft_Ny,fft_Nx));
    end
  end      
  C = abs(C);

  if ~isempty(shape)
    switch shape
      case 'same'
        y_range = round(fft_Ny/2 - size(A,1)/2)+1:round(fft_Ny/2 + size(A,1)/2);
        x_range = round(fft_Nx/2 - size(A,2)/2)+1:round(fft_Nx/2 + size(A,2)/2);
        C = C(y_range,x_range,:,:,:);
    end
  end

end