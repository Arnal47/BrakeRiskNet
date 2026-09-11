import csv
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 plt.figure(figsize=(7,4))
 for model in ['mlp','gru']:
  with open(f'results/v2/{model}_history.csv',newline='',encoding='utf-8') as f:r=list(csv.DictReader(f));plt.plot([x['epoch'] for x in r],[x['validation_loss'] for x in r],label=f'{model.upper()} validation')
 plt.xlabel('Epoch');plt.ylabel('Multi-task validation loss');plt.legend();plt.grid(alpha=.3);plt.tight_layout();plt.savefig('results/v2/validation_loss_curve.png',dpi=160)
if __name__=='__main__':main()
