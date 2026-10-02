"""Plot saved acceptance evidence without collecting or training again."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path.cwd() / 'logs/ea2')
    ap.add_argument('--run-name', required=True)
    args = ap.parse_args()
    evidence = args.root / 'acceptance' / args.run_name
    pipeline = json.loads((evidence / 'pipeline.json').read_text())
    recipes = ['baseline', 'aux_safe']
    metrics, rollouts, provenance = {}, {}, {}
    for recipe in recipes:
        directory = args.root / 'runs' / f'{args.run_name}-{recipe}'
        metrics[recipe] = json.loads((directory / 'model_metrics.json').read_text())
        rollouts[recipe] = json.loads((directory / 'rollout_metrics.json').read_text())
        provenance[recipe] = json.loads((directory / 'sl_config.json').read_text())
    assert provenance['baseline']['data_files'] == provenance['aux_safe']['data_files'] == pipeline['data_sha256']
    assert provenance['baseline']['split'] == provenance['aux_safe']['split']
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .25})
    colors = {'baseline': '#0072B2', 'aux_safe': '#D55E00'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout='constrained')
    for recipe in recipes:
        history = metrics[recipe]['history']
        epochs = np.arange(1, len(history['train_mse']) + 1)
        axes[0].plot(epochs, history['train_mse'], 'o-', color=colors[recipe], label=recipe)
        axes[1].plot(epochs, history['val_mse'], 'o-', color=colors[recipe], label=recipe)
        axes[2].plot(epochs, history['val_r2'], 'o-', color=colors[recipe], label=recipe)
    for ax, title, ylabel in zip(axes, ['Training', 'Validation', 'Validation'], ['MSE', 'MSE', 'R²']):
        ax.set(title=title, xlabel='Epoch', ylabel=ylabel, xticks=epochs)
        ax.legend()
    axes[2].axhline(0, color='gray', linestyle='--', linewidth=1)
    fig.suptitle('EA2 migration: frozen 2-map corpus, 5 epochs per recipe')
    fig.savefig(evidence / 'learning_curves.png', dpi=160)
    plt.close(fig)

    modes = ['zero_action', 'constant_oracle_mean', 'sl_stateful', 'sl_window']
    labels = ['zero action', 'constant', 'stateful', 'window']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for i, recipe in enumerate(recipes):
        map_results = [v['rollouts'] for k, v in rollouts[recipe].items() if k.startswith('map_seed')]
        same = [np.mean([m[mode]['same_frame_mse'] for m in map_results]) for mode in modes[1:]]
        collision = [100 * np.mean([m[mode]['collision_rate'] for m in map_results]) for mode in modes]
        axes[0].bar(np.arange(3) + (i - .5) * .34, same, .34, color=colors[recipe], label=recipe)
        axes[1].bar(np.arange(4) + (i - .5) * .34, collision, .34, color=colors[recipe], label=recipe)
    axes[0].set(title='Same-frame prediction error', ylabel='MSE', xticks=np.arange(3), xticklabels=labels[1:])
    axes[1].set(title='Geometric collision frames', ylabel='Frame fraction (%)', xticks=np.arange(4), xticklabels=labels)
    for ax in axes:
        ax.legend()
        ax.grid(axis='x', visible=False)
    fig.suptitle('Mean of two maps; 8 environments × 700 steps/mode; first 30 steps excluded')
    fig.supxlabel('Modes run on successive reset trajectories. Window mode outputs 0.5 until 400-frame history fills.')
    fig.savefig(evidence / 'rollout_comparison.png', dpi=160)
    plt.close(fig)
    print(f'REPORT SAVED {evidence / "learning_curves.png"} and rollout_comparison.png')


if __name__ == '__main__':
    main()
