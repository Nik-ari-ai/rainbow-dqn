# Rainbow DQN — Space Invaders

Реализация: [Kaixhin/Rainbow](https://github.com/Kaixhin/Rainbow) (MIT, commit `1745b18`), перенесена в `src/` и переведена с `atari-py` на `ale-py`. Все 6 компонентов Rainbow: Double DQN, Prioritized Replay, Dueling, Multi-step (n=3), C51, Noisy Nets.

## Установка (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # должно быть True
```

## Обучение

```powershell
python src/main.py --id spaceinv --game space_invaders --T-max 2000000 --checkpoint-interval 100000 --evaluation-interval 50000 --memory results/spaceinv/memory.pth.bz2
```

`--T-max` считается в шагах агента (1 шаг = 4 кадра). Лучшая модель по оценке сохраняется в `results/<id>/model.pth`, метрики в `results/<id>/metrics.pth`, графики в `Reward.html`/`Q.html`.
Продолжить после остановки: добавить `--model results/spaceinv/checkpoint.pth`.

## Оценка

```powershell
python src/main.py --id eval --evaluate --model results/spaceinv/model.pth --evaluation-episodes 20
```
