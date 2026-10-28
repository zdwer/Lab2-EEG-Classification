from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import torch

from dataloader_v2 import build_loaders_v2
from main import (Experiment, seed_all, train_one, write_history,
                  save_json)
from plotting import plot_experiment_summary


@dataclass(frozen=True)
class NewExperiment:
    dropout: float = .25
    weight_decay: float = .001
    subject_norm: bool = False
    time_jitter: int = 0


VARIANTS = {
    'relu_wd0001': NewExperiment(weight_decay=.0001),
    'relu_dropout01': NewExperiment(dropout=.1),
    'relu_subjectnorm': NewExperiment(subject_norm=True),
    'relu_timejitter': NewExperiment(time_jitter=10),
    'relu_wd0': NewExperiment(weight_decay=0),
    'relu_dropout015': NewExperiment(dropout=.15),
    'relu_subjectnorm_wd0001': NewExperiment(weight_decay=.0001, subject_norm=True),
}
DEFAULT_VARIANTS = ('relu_wd0001','relu_dropout01', 'relu_subjectnorm', 'relu_timejitter')


def main():
    ap=argparse.ArgumentParser(description='Lab2 EEGNet V2 targeted improvements')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--output', default='results_v2')
    ap.add_argument('--experiments', nargs='+', choices=tuple(VARIANTS), default=list(DEFAULT_VARIANTS))
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--patience',type=int,default=25)
    ap.add_argument('--val-fraction',type=float,default=.15)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--num-workers',type=int,default=0)
    ap.add_argument('--no-amp',action='store_true')
    ap.add_argument('--no-epoch-test',action='store_true',help='Avoid per-epoch test diagnostics; original Lab2 plots already cover them')
    ap.add_argument('--cpu',action='store_true')
    ap.add_argument('--smoke-test',action='store_true')
    args=ap.parse_args()
    if not 0 < args.val_fraction < .5 or args.epochs<1 or args.batch_size<1 or args.patience<0 or args.lr<=0:
        ap.error('Invalid argument: inspect train configuration')
    seed_all(args.seed)
    device=torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device)
    if args.smoke_test:
        args.epochs=1
        args.patience=0
        if args.output == 'results_v2':
            args.output='smoke_v2'
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'run_config.json',{
        'run_args': vars(args),
        'variants': {name: asdict(VARIANTS[name]) for name in args.experiments},
        'model_selection': 'Highest validation accuracy; validation loss breaks ties.',
        'test_warning': 'Previously inspected TEST results mean follow-up is exploratory, not an untouched holdout.'
    })
    rows=[]
    for name in args.experiments:
        variant=VARIANTS[name]
        loaders,counts=build_loaders_v2(
            args.data_dir,batch_size=args.batch_size,val_fraction=args.val_fraction,
            seed=args.seed,num_workers=args.num_workers,
            pin_memory=device.type=='cuda', subject_norm=variant.subject_norm,
            time_jitter=variant.time_jitter,
        )
        args.weight_decay=variant.weight_decay
        cfg=Experiment(model='eegnet',activation='relu',dropout=variant.dropout)
        result=train_one(name,cfg,loaders,args,device)
        val=result['final_validation']
        test=result['final_test']
        rows.append({
            'experiment':name, 'model':'eegnet','activation':'relu',
            'elu_alpha':1.0,'dropout':variant.dropout,
            'weight_decay':variant.weight_decay, 'subject_norm':variant.subject_norm,
            'time_jitter':variant.time_jitter,
            'selected_epoch':result['selected_epoch_by_validation_accuracy'],
            'validation_accuracy':val['accuracy'],
            'final_test_accuracy':test['accuracy'],
            'final_test_f1':test['f1'],
            'observed_peak_test_accuracy_diagnostic':result['highest_observed_epoch_test_accuracy_diagnostic'],
            'train_seconds':result['train_seconds'],
        })
    write_history(output/'experiment_summary.csv',rows)
    plot_experiment_summary(rows,output)
    selected=sorted(rows,key=lambda x:(-x['validation_accuracy'], x['experiment']))[0]
    save_json(output/'best_config.json',{
        'selected_experiment':selected['experiment'],
        'validation_accuracy':selected['validation_accuracy'],
        'selection_policy':'Validation only; NOT chosen by test accuracy.',
        'note':'Additional tests are exploratory: test set previously examined.'
    })
    print('\nALL V2 EXPERIMENTS COMPLETE')
    print('Validation-selected experiment:',selected['experiment'])
    print('Results:', output.resolve())

if __name__=='__main__':
    main()
