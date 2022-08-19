from .unet_convnext import UnetConvNextBlock
from .unet_resnet import UnetResNetBlock

def get_model(args, with_time_emb=True):
    if args.model == 'UnetConvNext':
        return UnetConvNextBlock(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels=3,
            with_time_emb=with_time_emb,
            residual=False,
        )
    if args.model == 'UnetResNet':
        if 'cifar10' in args.dataset:
            return UnetResNetBlock(resolution=32,
                in_channels=3,
                out_ch=3,
                ch=128,
                ch_mult=(1,2,2,2),
                num_res_blocks=2,
                attn_resolutions=(16,),
                with_time_emb=with_time_emb,
                dropout=0.1
            ) 
        if 'celebA' in args.dataset:
            return UnetResNetBlock(resolution=128,
                in_channels=3,
                out_ch=3,
                ch=128,
                ch_mult=(1,2,2,2),
                num_res_blocks=2,
                attn_resolutions=(16,),
                with_time_emb=with_time_emb,
                dropout=0.1
            ) 
 
