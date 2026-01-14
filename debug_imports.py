
try:
    from isaaclab.utils.math import subtract_frame_transforms
    print("Found in isaaclab.utils.math")
except ImportError:
    print("Not found in isaaclab.utils.math")

try:
    import isaacsim.core.utils.torch as torch_utils
    print("Imported torch_utils")
    if hasattr(torch_utils, 'subtract_frame_transforms'):
         print("Found in torch_utils")
    else:
         print("Not found in torch_utils")
except ImportError:
    print("torch_utils import failed")
