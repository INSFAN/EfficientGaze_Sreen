import yaml
import numpy as np

with open('pred_gt7.yml', 'r', encoding="utf-8") as f:
    yml_data = f.read()
    pred_gt = yaml.load(yml_data)
pred = pred_gt['pred']
gt = pred_gt['gt']

print(type(pred))

pred = np.array(pred).reshape(-1, 2)
gt = np.array(gt).reshape(-1, 2)

res = np.mean(np.sqrt(np.sum((pred-gt)*(pred-gt), axis=1)))

print(res)
