% 20210417 by Dushan N. Wadduwage
% This will perform 3d convolution comparable to Matlab's convn (with n=3),but for a
% batch of data in the form of an N-d array (N=4or5). The convolution is be
% done on the first two dimentions. f_conv3nd uses fftn for enhanced speed.
%
% inputs: A : 5d array
%         B : 3d array
% output  C : 5d array

function C = f_conv3nd(A,B,shape)

  fft_Ny      = size(A,1)+size(B,1)-1;
  fft_Nx      = size(A,2)+size(B,2)-1;
  fft_Nz      = size(A,3)+size(B,3)-1;
  
  for i=1:size(B,5)
    for j=1:size(B,4)
      %[i j]
      C(:,:,:,j,i) = ifftn(fftn(A,[fft_Ny,fft_Nx,fft_Nz]) .* fftn(B(:,:,:,j,i),[fft_Ny,fft_Nx,fft_Nz]));
    end
  end
  C = abs(C);

  if ~isempty(shape)
    switch shape
      case 'same'
        y_range = round(fft_Ny/2 - size(A,1)/2)+1:round(fft_Ny/2 + size(A,1)/2);
        x_range = round(fft_Nx/2 - size(A,2)/2)+1:round(fft_Nx/2 + size(A,2)/2);
        z_range = round(fft_Nz/2 - size(A,3)/2)+1:round(fft_Nz/2 + size(A,3)/2);
        C = C(y_range,x_range,z_range,:,:);
    end
  end

end