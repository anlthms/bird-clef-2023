import sys
import torch

model_file = sys.argv[1]

print(f'Loading model from {model_file}')
checkpoint = torch.load(model_file)
state = {
          'model': checkpoint['model'],
          'conf': checkpoint['conf']
         }
torch.save(state, model_file)
print(f'Stripped model saved to {model_file}')