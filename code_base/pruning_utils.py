import torch.nn.utils.prune as prune
import torch.nn as nn
import torch


# Apply structured pruning to the convolutional and fully connected layers
def apply_pruning(module, amount=0.3):
    """Apply structured pruning to convolutional and fully connected layers."""
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Conv2d):
            # Prune entire filters (along dim=0) based on L2 norm
            prune.ln_structured(layer, name='weight', amount=amount, n=2, dim=0)
            print(f"Applied structured pruning to Conv2d layer {name} with {amount * 100:.1f}% filters pruned.")
        elif isinstance(layer, nn.Linear):
            # Prune entire neurons (along dim=1) based on L2 norm
            prune.ln_structured(layer, name='weight', amount=amount, n=2, dim=1)
            print(f"Applied structured pruning to Linear layer {name} with {amount * 100:.1f}% neurons pruned.")
        elif name == "convGRU1.cell" or name == "convGRU2.cell" or name == "convGRU3.cell":
            prune.random_unstructured(layer, name="w_xz1", amount=amount)
            prune.random_unstructured(layer, name="w_xz2", amount=amount)
            prune.random_unstructured(layer, name="w_xz3", amount=amount)
            prune.random_unstructured(layer, name="w_hz", amount=amount)
            
            prune.random_unstructured(layer, name="w_xr1", amount=amount)
            prune.random_unstructured(layer, name="w_xr2", amount=amount)
            prune.random_unstructured(layer, name="w_xr3", amount=amount)
            prune.random_unstructured(layer, name="w_hr", amount=amount)
            
            prune.random_unstructured(layer, name="w_xh1", amount=amount)
            prune.random_unstructured(layer, name="w_xh2", amount=amount)
            prune.random_unstructured(layer, name="w_xh3", amount=amount)
            prune.random_unstructured(layer, name="w_hh", amount=amount)

            print(f"Applied structured pruning to convGRU layer {name} with {amount * 100:.1f}% neurons pruned.")
    # Remove pruning to finalize the reduced model
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            prune.remove(layer, 'weight')
        elif name == "convGRU1.cell" or name == "convGRU2.cell" or name == "convGRU3.cell":
            prune.remove(layer, "w_xz1")
            prune.remove(layer, "w_xz2")
            prune.remove(layer, "w_xz3")
            prune.remove(layer, "w_hz")
            
            prune.remove(layer, "w_xr1")
            prune.remove(layer, "w_xr2")
            prune.remove(layer, "w_xr3")
            prune.remove(layer, "w_hr")
            
            prune.remove(layer, "w_xh1")
            prune.remove(layer, "w_xh2")
            prune.remove(layer, "w_xh3")
            prune.remove(layer, "w_hh")
            
# Check how many non-zero weights remain after pruning
def print_nonzero_weights(model):
    """Print the percentage of remaining non-zero weights in the model."""
    total_params, nonzero_params = 0, 0
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' or 'w_xz1' in name:
            total_params += param.numel()
            nonzero_params += param.nonzero().size(0)
            
    print(f"Non-zero parameters: {nonzero_params}/{total_params} ({100 * nonzero_params / total_params:.2f}%)")





class MaskedAdam(torch.optim.Adam):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

    def step(self, closure=None):
        """Override the step function to skip updates for zeroed weights."""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                # Mask the gradients: set gradient to zero where parameter is zero
                grad_mask = param.data != 0  # Boolean mask: True where weights are non-zero
                param.grad.data.mul_(grad_mask)  # Zero-out gradients for zeroed weights

        # Call the original Adam step function
        super().step(closure)