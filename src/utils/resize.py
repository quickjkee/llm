import torch


def fft_downsample(img: torch.Tensor, size: int) -> torch.Tensor:
    img_dtype = img.dtype
    # fft doesn't support fp16 data types
    img = img.float()
    h, w = img.shape[-2:]
    assert h == w and h % 2 == 0
    
    cy, cx = h // 2, w // 2  # centerness
    r = size // 2
    
    fft_img = torch.fft.fft2(img, norm="forward")
    fft_shift_img = torch.fft.fftshift(fft_img)

    # Remove high freqs
    fft_shift_img = fft_shift_img[:, :, cy - r:cy + r, cx - r:cx + r]
    
    ifft_shift_img = torch.fft.ifftshift(fft_shift_img)
    ifft_img = torch.fft.ifft2(ifft_shift_img, norm="forward").real
    
    assert ifft_img.shape[-1] == ifft_img.shape[-2] == size
    return ifft_img.to(img_dtype)


def fft_upsample(img: torch.Tensor, size: int) -> torch.Tensor:
    img_dtype = img.dtype
    # fft doesn't support fp16 data types
    img = img.float()
    b, ch, h, w = img.shape
    assert h == w and h % 2 == 0
    
    cy, cx = size // 2, size // 2  # centerness
    r = w // 2

    fft_img = torch.fft.fft2(img, norm="forward")
    fft_shift_img = torch.fft.fftshift(fft_img)
    
    hr_fft_shift_img = torch.zeros(
        b, ch, size, size, 
        device=fft_shift_img.device, 
        dtype=fft_shift_img.dtype
    )

    # Copy low freqs
    hr_fft_shift_img[:, :, cy - r:cy + r, cx - r:cx + r] = fft_shift_img
    
    ifft_shift_img = torch.fft.ifftshift(hr_fft_shift_img)
    ifft_img = torch.fft.ifft2(ifft_shift_img, norm="forward").real
    assert ifft_img.shape[-1] == ifft_img.shape[-2] == size
    return ifft_img.to(img_dtype)
