from einops import rearrange
from mia.ops.naive_grid_sample import naive_grid_sample_1d, naive_grid_sample_2d

def grid_sample(input, coords):
    if coords.shape[-1] == 1:
        # 1D case (synthetic, audio)
        # input shape       (B, D, L)
        # coords shape      (B, N, 1)
        # output shape      (B, D, N)
        output = naive_grid_sample_1d(input, coords)
        output = rearrange(output, 'b d n -> b n d')
        return output

    elif coords.shape[-1] == 2:
        # 2D case (image)
        # input shape       (B, D, H, W)
        # coords shape      (B, N, 2)
        # coords -> grid    (B, 1, N, 2)
        grid = coords.unsqueeze(1)[..., [1, 0]]
        output = naive_grid_sample_2d(input, grid)
        output = rearrange(output, 'b d 1 n -> b n 1 d').squeeze(2)
        return output
    else:
        raise NotImplementedError('Only 1D, 2D cases are implemented.')