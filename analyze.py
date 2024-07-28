import sys
import pickle
import torch

class CustomUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        storage_type, key, location, numel = pid[1:]
        if storage_type is torch.UntypedStorage:
            dtype = torch.uint8
        else:
            dtype = storage_type.dtype

        nbytes = numel * torch._utils._element_size(dtype)
        storage = torch.UntypedStorage(nbytes, device="cpu")

        return torch.storage.TypedStorage(wrap_storage=storage, dtype=dtype)

with open('data.pkl', 'rb') as f:
    data = CustomUnpickler(f).load()

with open('layers', 'w') as f:
    for k, v in data.items():
        f.write(f'{k} {' '.join(str(s) for s in v.size())}\n')

print('done')
